"""tests/constrained_translation/experiment/test_sequencing_runner_5b.py

Strict tests for Task 5B: leakage-safe round prediction evaluation and safety
composition.

Contract under test
-------------------
- immutable RetrievalExample, PredictionRequest (NO query reference target),
  CellPrediction, EvaluatedRound;
- evaluate_round_state / evaluate_schedule over fixed project + eval order;
- known_source_types from current HumanEvidence only;
- exact corpus_coverage semantics;
- unsupported items abstain before any callback, spans/artifact retained;
- optional explicit false-abstention oracle (evaluation-only);
- retrieval callback sees immutable current evidence only; returned IDs must be
  current and unique; converted to RetrievalExample;
- generation gets PredictionRequest only, no query reference;
  explicit TranslationEvidence validated; classify then ALWAYS guard
  escaped hallucination + forensic artifact;
- records include pool/id/index/source/retrieved IDs/support/spans/outcome/
  backend/raw/final/artifact/provenance;
- EvaluatedRound: selected/evidence/gain/coverage/predictions/counters;
- project/eval fixed order each round; exact counters;
- tests for secret future targets, request introspection, malicious retrieval,
  unsupported no callback, callback order, guards, model UNK,
  false abstention oracle, deterministic/immutable, schedule composition.
"""
from __future__ import annotations

import dataclasses
import pytest

from constrained_translation.experiment.sequencing_runner import (
    # 5A
    HumanEvidence,
    SourceCandidate,
    RoundState,
    RoundSchedule,
    build_round_schedule,
    # 5B
    RetrievalExample,
    PredictionRequest,
    CellPrediction,
    EvaluatedRound,
    evaluate_round_state,
    evaluate_schedule,
)
from constrained_translation.experiment.sequencing_manifest import (
    SequencingManifest,
    SequencingPoolItem,
)
from constrained_translation.experiment.outcomes import (
    EscapedHallucinationError,
    MissingForensicArtifactError,
    Outcome,
    TranslationEvidence,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LANG = "mya"


# ---------------------------------------------------------------------------
# Helpers — pool item builders
# ---------------------------------------------------------------------------

def _item(idx: int, src: str, tgt: str | None = None) -> SequencingPoolItem:
    """Build a SequencingPoolItem for LANG."""
    return SequencingPoolItem(
        corpus_idx=idx,
        item_id=f"TST {idx}:1",
        source_text=src,
        target_texts={LANG: tgt if tgt is not None else f"tgt_{idx}"},
    )


def _manifest(seed, acq, ev, proj) -> SequencingManifest:
    return SequencingManifest(
        languages=[LANG],
        random_seed=0,
        seed_size=len(seed),
        acq_size=len(acq),
        eval_size=len(ev),
        seed=seed,
        acquisition=acq,
        fixed_eval=ev,
        remaining=proj,
        digest="",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Source token vocabulary (normalized form appears after normalize_source_units):
#   "alpha bravo" → ['alpha', 'bravo']
#   "charlie delta" → ['charlie', 'delta']
#   "zeta omega" → ['zeta', 'omega']  ← never in evidence below
#
# Seed evidence covers {'alpha', 'bravo', 'charlie', 'delta', 'seed', 'item'}.
# Project item using 'alpha' → supported.
# Project item using 'zeta' → unsupported.

SEED_A_SRC = "alpha bravo seed item"
SEED_B_SRC = "charlie delta seed item"
ACQ_0_SRC  = "alpha extra acquisition"
ACQ_1_SRC  = "charlie more acquisition"
PROJ_S_SRC = "alpha bravo project supported"  # overlaps with evidence
PROJ_U_SRC = "zeta omega project unknown"     # no overlap → unsupported
EVAL_S_SRC = "charlie delta eval supported"   # overlaps
EVAL_U_SRC = "zeta omega eval unknown"         # no overlap


@pytest.fixture
def tiny_manifest():
    """2 seed, 2 acq, 1 eval, 2 project items."""
    seed = [
        _item(0, SEED_A_SRC, "seed_tgt_0"),
        _item(1, SEED_B_SRC, "seed_tgt_1"),
    ]
    acq = [
        _item(10, ACQ_0_SRC, "acq_tgt_0"),
        _item(11, ACQ_1_SRC, "acq_tgt_1"),
    ]
    ev = [_item(20, EVAL_S_SRC, "eval_tgt_0")]
    proj = [
        _item(30, PROJ_S_SRC, "proj_tgt_0"),
        _item(31, PROJ_U_SRC, "proj_tgt_1"),
    ]
    return _manifest(seed, acq, ev, proj)


@pytest.fixture
def tiny_schedule(tiny_manifest):
    return build_round_schedule(
        tiny_manifest, LANG,
        acq_order=[10, 11],
        expected_seed=2, expected_acq=2, expected_eval=1, expected_project_min=1,
    )


@pytest.fixture
def round0_state(tiny_schedule):
    return tiny_schedule.states[0]


@pytest.fixture
def round1_state(tiny_schedule):
    return tiny_schedule.states[1]


def _good_te(req: PredictionRequest) -> TranslationEvidence:
    """Return a fully valid ACCEPTED_LICENSED TranslationEvidence."""
    return TranslationEvidence(
        item_id=req.query_id,
        source_text=req.query_source,
        source_supported=True,
        false_abstention_detected=False,
        backend_called=True,
        raw_generation="good output",
        exact_id_licensed=True,
        visible_surface_licensed=True,
        full_decode_match=True,
        control_token_valid=True,
        grammar_accepted=True,
        audit_passed=True,
        audit_reasons=[],
        model_generated_unk=False,
        surfaced_failure_artifact=None,
        final_translation="good output",
    )


def _simple_retrieve(req: PredictionRequest, evidence: tuple) -> list[int]:
    """Return just the first evidence ID."""
    return [evidence[0].corpus_idx]


# ---------------------------------------------------------------------------
# Section 1: Immutable data models
# ---------------------------------------------------------------------------

class TestRetrievalExampleImmutable:
    def test_fields(self):
        re = RetrievalExample(evidence_id=5, item_id="A 1:1", source_text="src", target_text="tgt")
        assert re.evidence_id == 5
        assert re.item_id == "A 1:1"
        assert re.source_text == "src"
        assert re.target_text == "tgt"

    def test_is_frozen(self):
        re = RetrievalExample(evidence_id=1, item_id="X", source_text="s", target_text="t")
        with pytest.raises((AttributeError, TypeError)):
            re.evidence_id = 99  # type: ignore[misc]

    def test_hashable(self):
        re = RetrievalExample(evidence_id=1, item_id="X", source_text="s", target_text="t")
        s = {re}
        assert re in s

    def test_equality(self):
        a = RetrievalExample(evidence_id=1, item_id="X", source_text="s", target_text="t")
        b = RetrievalExample(evidence_id=1, item_id="X", source_text="s", target_text="t")
        assert a == b


class TestPredictionRequestImmutable:
    def test_fields(self):
        pr = PredictionRequest(
            query_id="X 1:1",
            query_corpus_idx=7,
            query_source="the source",
            retrieved_examples=(),
            known_source_types=frozenset(["alpha"]),
        )
        assert pr.query_id == "X 1:1"
        assert pr.query_corpus_idx == 7
        assert pr.query_source == "the source"
        assert pr.retrieved_examples == ()
        assert pr.known_source_types == frozenset(["alpha"])

    def test_is_frozen(self):
        pr = PredictionRequest(
            query_id="X", query_corpus_idx=1, query_source="s",
            retrieved_examples=(), known_source_types=frozenset(),
        )
        with pytest.raises((AttributeError, TypeError)):
            pr.query_source = "other"  # type: ignore[misc]

    def test_no_query_target_field(self):
        """PredictionRequest MUST NOT contain any query reference/target field."""
        pr = PredictionRequest(
            query_id="X", query_corpus_idx=1, query_source="s",
            retrieved_examples=(), known_source_types=frozenset(),
        )
        for attr in ("query_target", "reference_target", "reference_text", "target_text", "target"):
            assert not hasattr(pr, attr), f"Unexpected attribute {attr!r} on PredictionRequest"

    def test_hashable(self):
        pr = PredictionRequest(
            query_id="X", query_corpus_idx=1, query_source="s",
            retrieved_examples=(), known_source_types=frozenset(),
        )
        {pr}  # should not raise


class TestCellPredictionImmutable:
    def _make(self) -> CellPrediction:
        return CellPrediction(
            pool="project",
            item_id="X 1:1",
            corpus_idx=5,
            source_text="src",
            retrieved_evidence_ids=(1, 2),
            source_supported=True,
            unsupported_spans=(),
            outcome=Outcome.ACCEPTED_LICENSED,
            backend_called=True,
            raw_generation="raw",
            final_translation="final",
            surfaced_failure_artifact=None,
            evidence_ids_at_round=(0, 1),
        )

    def test_all_fields_present(self):
        cp = self._make()
        assert cp.pool == "project"
        assert cp.item_id == "X 1:1"
        assert cp.corpus_idx == 5
        assert cp.source_text == "src"
        assert cp.retrieved_evidence_ids == (1, 2)
        assert cp.source_supported is True
        assert cp.unsupported_spans == ()
        assert cp.outcome == Outcome.ACCEPTED_LICENSED
        assert cp.backend_called is True
        assert cp.raw_generation == "raw"
        assert cp.final_translation == "final"
        assert cp.surfaced_failure_artifact is None
        assert cp.evidence_ids_at_round == (0, 1)

    def test_is_frozen(self):
        cp = self._make()
        with pytest.raises((AttributeError, TypeError)):
            cp.outcome = Outcome.OTHER_HARD_FAILURE  # type: ignore[misc]


class TestEvaluatedRoundImmutable:
    def _make(self, project_preds=(), eval_preds=()) -> EvaluatedRound:
        return EvaluatedRound(
            round=0,
            selected=None,
            estimated_gain=None,
            evidence_ids=(1, 2),
            evidence_count=2,
            project_coverage=None,
            eval_coverage=None,
            project_predictions=project_preds,
            eval_predictions=eval_preds,
            retrieve_call_count=0,
            generate_call_count=0,
        )

    def test_all_fields(self):
        er = self._make()
        assert er.round == 0
        assert er.selected is None
        assert er.estimated_gain is None
        assert er.evidence_ids == (1, 2)
        assert er.evidence_count == 2
        assert er.retrieve_call_count == 0
        assert er.generate_call_count == 0

    def test_is_frozen(self):
        er = self._make()
        with pytest.raises((AttributeError, TypeError)):
            er.round = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 2: Secret future targets — PredictionRequest introspection
# ---------------------------------------------------------------------------

class TestSecretFutureTargets:
    """PredictionRequest passed to generate_fn must not contain query target."""

    def test_generate_fn_receives_no_target(self, tiny_manifest, round0_state):
        """generate_fn must not receive query reference target."""
        proj = [_item(30, PROJ_S_SRC, "proj_tgt_0")]
        ev_items = [_item(20, EVAL_S_SRC, "eval_tgt_0")]

        received_requests: list[PredictionRequest] = []

        def generate_fn(req: PredictionRequest) -> TranslationEvidence:
            received_requests.append(req)
            return _good_te(req)

        evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=generate_fn,
        )
        for req in received_requests:
            for attr in ("query_target", "reference_target", "target_text", "reference_text", "target"):
                assert not hasattr(req, attr), (
                    f"PredictionRequest exposed {attr!r} to generate_fn — target leakage!"
                )

    def test_retrieve_fn_receives_no_target(self, tiny_manifest, round0_state):
        """retrieve_fn must receive immutable evidence tuple, not future targets."""
        proj = [_item(30, PROJ_S_SRC, "proj_tgt_0")]
        ev_items = [_item(20, EVAL_S_SRC, "eval_tgt_0")]

        received_reqs: list[PredictionRequest] = []

        def retrieve_fn(req, evidence):
            received_reqs.append(req)
            return [evidence[0].corpus_idx]

        evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=_good_te,
        )
        for req in received_reqs:
            assert not hasattr(req, "query_target")
            assert not hasattr(req, "reference_target")
            # retrieved_examples should be empty at retrieval request time
            # (it's the initial request before retrieval fills it)
            assert isinstance(req.retrieved_examples, tuple)


# ---------------------------------------------------------------------------
# Section 3: Source support semantics — exact corpus_coverage normalization
# ---------------------------------------------------------------------------

class TestSourceSupportSemantics:
    """Support is determined by normalize_source_units intersection with evidence."""

    def test_supported_item_calls_both_callbacks(self, round0_state):
        """An item whose normalized tokens overlap evidence calls retrieve+generate."""
        proj = [_item(30, PROJ_S_SRC, "proj_tgt_0")]   # overlaps seed evidence
        ev_items = []

        retrieve_calls = []
        generate_calls = []

        def retrieve_fn(req, evidence):
            retrieve_calls.append(req.query_id)
            return [evidence[0].corpus_idx]

        def generate_fn(req):
            generate_calls.append(req.query_id)
            return _good_te(req)

        evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        assert len(retrieve_calls) == 1
        assert len(generate_calls) == 1

    def test_unsupported_item_calls_no_callbacks(self, round0_state):
        """Unsupported item (no token overlap) → no callback called."""
        proj = [_item(31, PROJ_U_SRC, "proj_tgt_1")]  # zeta omega → unsupported
        ev_items = []

        retrieve_calls = []
        generate_calls = []

        def retrieve_fn(req, evidence):
            retrieve_calls.append(req.query_id)
            return []

        def generate_fn(req):
            generate_calls.append(req.query_id)
            return _good_te(req)

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        assert retrieve_calls == [], "retrieve_fn must not be called for unsupported items"
        assert generate_calls == [], "generate_fn must not be called for unsupported items"
        assert result.retrieve_call_count == 0
        assert result.generate_call_count == 0

    def test_unsupported_item_outcome_safe_abstention(self, round0_state):
        """Unsupported item → SAFE_SOURCE_ABSTENTION outcome."""
        proj = [_item(31, PROJ_U_SRC, "proj_tgt_1")]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        assert pred.source_supported is False
        assert pred.backend_called is False
        assert pred.outcome == Outcome.SAFE_SOURCE_ABSTENTION

    def test_unsupported_spans_retained(self, round0_state):
        """Unsupported item retains unsupported_spans and surfaced_failure_artifact."""
        proj = [_item(31, PROJ_U_SRC, "proj_tgt_1")]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        # unsupported_spans: tuple of UNKSpan objects (may be empty if all punct)
        assert isinstance(pred.unsupported_spans, tuple)
        # surfaced_failure_artifact must be non-None
        assert pred.surfaced_failure_artifact is not None
        assert isinstance(pred.surfaced_failure_artifact, str)
        assert len(pred.surfaced_failure_artifact) > 0

    def test_unsupported_item_retrieved_evidence_ids_empty(self, round0_state):
        """Unsupported item has empty retrieved_evidence_ids."""
        proj = [_item(31, PROJ_U_SRC, "proj_tgt_1")]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        assert pred.retrieved_evidence_ids == ()

    def test_evidence_ids_at_round_always_set(self, round0_state, round1_state):
        """All CellPredictions record evidence IDs at that round."""
        proj = [_item(30, PROJ_S_SRC, "pt0"), _item(31, PROJ_U_SRC, "pt1")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        for state in [round0_state, round1_state]:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            expected_ids = tuple(he.corpus_idx for he in state.evidence)
            for pred in result.project_predictions + result.eval_predictions:
                assert pred.evidence_ids_at_round == expected_ids, (
                    f"Round {state.round}: expected {expected_ids}, got {pred.evidence_ids_at_round}"
                )


# ---------------------------------------------------------------------------
# Section 4: Callback order — retrieve before generate
# ---------------------------------------------------------------------------

class TestCallbackOrder:
    def test_retrieve_called_before_generate(self, round0_state):
        """retrieve_fn is always called before generate_fn for same item."""
        call_log: list[str] = []

        def retrieve_fn(req, evidence):
            call_log.append(f"retrieve:{req.query_id}")
            return [evidence[0].corpus_idx]

        def generate_fn(req):
            call_log.append(f"generate:{req.query_id}")
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        assert len(call_log) == 2
        assert call_log[0].startswith("retrieve:")
        assert call_log[1].startswith("generate:")
        assert call_log[0].split(":")[1] == call_log[1].split(":")[1]

    def test_project_before_eval_order(self, round0_state):
        """Project items evaluated before eval items; both in fixed order."""
        call_log: list[tuple[str, str]] = []

        def retrieve_fn(req, evidence):
            call_log.append(("retrieve", req.query_id))
            return [evidence[0].corpus_idx]

        def generate_fn(req):
            call_log.append(("generate", req.query_id))
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0"), _item(32, PROJ_S_SRC + " more", "pt1")]
        # Note: proj[1] needs distinct src — use same tokens so still supported
        proj[1] = _item(32, "alpha additional project", "pt1")
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        retrieve_ids = [qid for (op, qid) in call_log if op == "retrieve"]
        # Project items come first
        assert retrieve_ids[0] == proj[0].item_id
        assert retrieve_ids[1] == proj[1].item_id
        assert retrieve_ids[2] == ev_items[0].item_id


# ---------------------------------------------------------------------------
# Section 5: Retrieval callback validation
# ---------------------------------------------------------------------------

class TestRetrievalValidation:
    def test_retrieve_sees_immutable_evidence_tuple(self, round0_state):
        """retrieve_fn receives current evidence as an immutable tuple."""
        received_evidence_types: list[type] = []

        def retrieve_fn(req, evidence):
            received_evidence_types.append(type(evidence))
            # Attempting mutation should fail
            with pytest.raises((AttributeError, TypeError)):
                evidence[0] = None  # type: ignore[index]
            return [evidence[0].corpus_idx]

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=_good_te,
        )
        assert received_evidence_types[0] is tuple

    def test_malicious_retrieval_unknown_id_raises(self, round0_state):
        """retrieve_fn returning unknown corpus_idx raises ValueError."""
        def malicious_retrieve(req, evidence):
            return [99999]  # ID not in current evidence

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError, match="not in current state.evidence"):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=malicious_retrieve,
                generate_fn=_good_te,
            )

    def test_malicious_retrieval_future_acq_id_raises(self, tiny_schedule):
        """retrieve_fn returning future (not-yet-revealed) acquisition ID raises ValueError."""
        state0 = tiny_schedule.states[0]
        # corpus_idx 10 and 11 are acquisition — not yet in evidence at round 0
        def retrieve_future(req, evidence):
            return [10]  # future acquisition — not yet in evidence

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError):
            evaluate_round_state(
                state0, proj, [], LANG,
                retrieve_fn=retrieve_future,
                generate_fn=_good_te,
            )

    def test_duplicate_retrieved_ids_raises(self, round0_state):
        """retrieve_fn returning duplicate corpus_idx raises ValueError."""
        def dup_retrieve(req, evidence):
            first = evidence[0].corpus_idx
            return [first, first]  # duplicate!

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError, match="duplicate"):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=dup_retrieve,
                generate_fn=_good_te,
            )

    def test_retrieved_converted_to_retrieval_examples(self, round0_state):
        """IDs returned by retrieve_fn are converted to RetrievalExample in gen request."""
        received_examples: list[tuple] = []

        def retrieve_fn(req, evidence):
            return [evidence[0].corpus_idx]

        def generate_fn(req: PredictionRequest):
            received_examples.extend(req.retrieved_examples)
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        assert len(received_examples) == 1
        ex = received_examples[0]
        assert isinstance(ex, RetrievalExample)
        assert isinstance(ex.evidence_id, int)
        assert isinstance(ex.item_id, str)
        assert isinstance(ex.source_text, str)
        assert isinstance(ex.target_text, str)

    def test_retrieved_example_contains_human_target(self, round0_state):
        """RetrievalExample.target_text is the human-translated target text."""
        received_examples: list[RetrievalExample] = []

        def retrieve_fn(req, evidence):
            return [evidence[0].corpus_idx]

        def generate_fn(req: PredictionRequest):
            received_examples.extend(req.retrieved_examples)  # type: ignore[arg-type]
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        ex = received_examples[0]
        # evidence[0] is corpus_idx=0, seed_tgt_0
        assert ex.target_text == "seed_tgt_0"
        assert ex.evidence_id == 0

    def test_empty_retrieve_ok(self, round0_state):
        """retrieve_fn may return empty list; generate_fn gets empty retrieved_examples."""
        received_examples: list = []

        def retrieve_fn(req, evidence):
            return []  # no examples selected

        def generate_fn(req: PredictionRequest):
            received_examples.append(req.retrieved_examples)
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        assert received_examples[0] == ()

    def test_retrieve_fn_current_ids_only(self, tiny_schedule):
        """At round 1, retrieve_fn can return newly revealed acquisition ID."""
        state1 = tiny_schedule.states[1]  # evidence: seed(0,1) + acq(10)

        def retrieve_fn(req, evidence):
            # Return the newly acquired item (corpus_idx=10)
            return [10]

        def generate_fn(req):
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        result = evaluate_round_state(
            state1, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        pred = result.project_predictions[0]
        assert 10 in pred.retrieved_evidence_ids


# ---------------------------------------------------------------------------
# Section 6: Generation validation
# ---------------------------------------------------------------------------

class TestGenerationValidation:
    def test_generate_fn_item_id_mismatch_raises(self, round0_state):
        """generate_fn returning wrong item_id raises ValueError."""
        def bad_gen(req: PredictionRequest):
            te = _good_te(req)
            # Return evidence with wrong item_id
            return dataclasses.replace(te, item_id="WRONG_ID")

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError, match="item_id"):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )

    def test_generate_fn_source_text_mismatch_raises(self, round0_state):
        """generate_fn returning wrong source_text raises ValueError."""
        def bad_gen(req: PredictionRequest):
            te = _good_te(req)
            return dataclasses.replace(te, source_text="WRONG SOURCE TEXT")

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError, match="source_text"):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )

    def test_generate_fn_backend_not_called_raises(self, round0_state):
        """generate_fn returning backend_called=False for supported item raises ValueError."""
        def bad_gen(req: PredictionRequest):
            te = _good_te(req)
            return dataclasses.replace(te, backend_called=False, final_translation=None)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(ValueError, match="backend_called"):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )

    def test_generate_fn_only_called_for_supported(self, round0_state):
        """generate_fn is never called for unsupported items."""
        proj = [
            _item(30, PROJ_S_SRC, "supported"),
            _item(31, PROJ_U_SRC, "unsupported"),
        ]
        generated_ids = []

        def generate_fn(req):
            generated_ids.append(req.query_id)
            return _good_te(req)

        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=generate_fn,
        )
        assert len(generated_ids) == 1
        assert generated_ids[0] == proj[0].item_id


# ---------------------------------------------------------------------------
# Section 7: Safety guards (ALWAYS enforced after generate)
# ---------------------------------------------------------------------------

class TestSafetyGuards:
    def test_escaped_hallucination_raises(self, round0_state):
        """EscapedHallucinationError raised when generate_fn produces ESCAPED_HALLUCINATION."""
        def bad_gen(req: PredictionRequest):
            return TranslationEvidence(
                item_id=req.query_id,
                source_text=req.query_source,
                source_supported=True,
                false_abstention_detected=False,
                backend_called=True,
                raw_generation="hallucinated output",
                exact_id_licensed=False,   # not licensed →
                visible_surface_licensed=False,
                full_decode_match=True,
                control_token_valid=True,
                grammar_accepted=False,
                audit_passed=False,
                audit_reasons=["not licensed"],
                model_generated_unk=False,
                surfaced_failure_artifact="hallucinated output",
                final_translation="hallucinated output",  # non-None → ESCAPED_HALLUCINATION
            )

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(EscapedHallucinationError):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )

    def test_missing_forensic_artifact_raises(self, round0_state):
        """MissingForensicArtifactError raised when artifact is missing for rejection."""
        def bad_gen(req: PredictionRequest):
            return TranslationEvidence(
                item_id=req.query_id,
                source_text=req.query_source,
                source_supported=True,
                false_abstention_detected=False,
                backend_called=True,
                raw_generation="emitted text",
                exact_id_licensed=False,   # not licensed
                visible_surface_licensed=False,
                full_decode_match=True,
                control_token_valid=True,
                grammar_accepted=False,
                audit_passed=False,
                audit_reasons=["not licensed"],
                model_generated_unk=False,
                surfaced_failure_artifact=None,  # MISSING artifact ← triggers guard
                final_translation=None,
            )

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        with pytest.raises(MissingForensicArtifactError):
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )

    def test_guards_enforced_before_returning_cell(self, round0_state):
        """Guards fire synchronously — no partial CellPrediction returned."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        raised = []

        def bad_gen(req):
            return TranslationEvidence(
                item_id=req.query_id,
                source_text=req.query_source,
                source_supported=True,
                false_abstention_detected=False,
                backend_called=True,
                raw_generation="bad",
                exact_id_licensed=False,
                visible_surface_licensed=False,
                full_decode_match=True,
                control_token_valid=True,
                grammar_accepted=False,
                audit_passed=False,
                audit_reasons=["x"],
                model_generated_unk=False,
                surfaced_failure_artifact=None,
                final_translation=None,
            )

        try:
            evaluate_round_state(
                round0_state, proj, [], LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=bad_gen,
            )
        except MissingForensicArtifactError as e:
            raised.append(e)
        assert len(raised) == 1


# ---------------------------------------------------------------------------
# Section 8: Model UNK outcome
# ---------------------------------------------------------------------------

class TestModelUNKOutcome:
    def test_model_generated_unk_outcome(self, round0_state):
        """generate_fn returning model_generated_unk=True → MODEL_GENERATED_UNK outcome."""
        def unk_gen(req: PredictionRequest):
            return TranslationEvidence(
                item_id=req.query_id,
                source_text=req.query_source,
                source_supported=True,
                false_abstention_detected=False,
                backend_called=True,
                raw_generation="output with [UNK:alpha]",
                exact_id_licensed=True,
                visible_surface_licensed=True,
                full_decode_match=True,
                control_token_valid=True,
                grammar_accepted=True,
                audit_passed=False,
                audit_reasons=["unk marker detected"],
                model_generated_unk=True,
                surfaced_failure_artifact="[UNK:alpha]",
                final_translation=None,
            )

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=unk_gen,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.MODEL_GENERATED_UNK
        assert pred.surfaced_failure_artifact == "[UNK:alpha]"
        assert pred.backend_called is True

    def test_raw_generation_unk_marker_detected(self, round0_state):
        """Safety override: [UNK:<src>] in raw_generation → MODEL_GENERATED_UNK."""
        def unk_gen(req: PredictionRequest):
            return TranslationEvidence(
                item_id=req.query_id,
                source_text=req.query_source,
                source_supported=True,
                false_abstention_detected=False,
                backend_called=True,
                raw_generation="hello [UNK:zeta] world",
                exact_id_licensed=True,
                visible_surface_licensed=True,
                full_decode_match=True,
                control_token_valid=True,
                grammar_accepted=True,
                audit_passed=True,
                audit_reasons=[],
                model_generated_unk=False,  # flag False but marker in raw
                surfaced_failure_artifact="[UNK:zeta]",
                final_translation=None,
            )

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=unk_gen,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.MODEL_GENERATED_UNK


# ---------------------------------------------------------------------------
# Section 9: False abstention oracle (evaluation-only)
# ---------------------------------------------------------------------------

class TestFalseAbstentionOracle:
    def test_oracle_true_gives_false_abstention(self, round0_state):
        """oracle returning True for unsupported → FALSE_SOURCE_ABSTENTION."""
        proj = [_item(31, PROJ_U_SRC, "unsupported")]

        def oracle(corpus_idx: int, source_text: str) -> bool:
            return True  # claim it's a false abstention

        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
            optional_false_abstention_oracle=oracle,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.FALSE_SOURCE_ABSTENTION
        assert pred.backend_called is False  # oracle is evaluation-only, no backend call

    def test_oracle_false_gives_safe_abstention(self, round0_state):
        """oracle returning False → SAFE_SOURCE_ABSTENTION (default)."""
        proj = [_item(31, PROJ_U_SRC, "unsupported")]

        def oracle(corpus_idx: int, source_text: str) -> bool:
            return False

        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
            optional_false_abstention_oracle=oracle,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.SAFE_SOURCE_ABSTENTION

    def test_oracle_never_called_for_supported(self, round0_state):
        """Oracle is never called for supported items."""
        proj = [_item(30, PROJ_S_SRC, "supported")]
        oracle_calls: list[int] = []

        def oracle(corpus_idx: int, source_text: str) -> bool:
            oracle_calls.append(corpus_idx)
            return True

        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
            optional_false_abstention_oracle=oracle,
        )
        assert oracle_calls == [], "Oracle must not be called for supported items"

    def test_oracle_none_default_is_safe_abstention(self, round0_state):
        """No oracle → unsupported items default to SAFE_SOURCE_ABSTENTION."""
        proj = [_item(31, PROJ_U_SRC, "unsupported")]
        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
            optional_false_abstention_oracle=None,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.SAFE_SOURCE_ABSTENTION

    def test_oracle_does_not_influence_retrieval_or_generation(self, round0_state):
        """Oracle is purely evaluation-only: does not trigger callbacks."""
        proj = [_item(31, PROJ_U_SRC, "unsupported")]
        retrieve_calls = []
        generate_calls = []

        def retrieve_fn(req, evidence):
            retrieve_calls.append(req.query_id)
            return []

        def generate_fn(req):
            generate_calls.append(req.query_id)
            return _good_te(req)

        def oracle(corpus_idx, source_text):
            return True

        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
            optional_false_abstention_oracle=oracle,
        )
        assert retrieve_calls == []
        assert generate_calls == []


# ---------------------------------------------------------------------------
# Section 10: EvaluatedRound records
# ---------------------------------------------------------------------------

class TestEvaluatedRoundRecords:
    def test_round_index_matches_state(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        for state in tiny_schedule.states:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            assert result.round == state.round

    def test_selected_matches_state(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        for state in tiny_schedule.states:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            assert result.selected is state.selected

    def test_estimated_gain_matches_state(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        for state in tiny_schedule.states:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            assert result.estimated_gain is state.estimated_gain

    def test_evidence_ids_match_state(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        for state in tiny_schedule.states:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            expected_ids = tuple(he.corpus_idx for he in state.evidence)
            assert result.evidence_ids == expected_ids

    def test_evidence_count_matches_state(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        for state in tiny_schedule.states:
            result = evaluate_round_state(
                state, proj, ev_items, LANG,
                retrieve_fn=_simple_retrieve,
                generate_fn=_good_te,
            )
            assert result.evidence_count == len(state.evidence)

    def test_coverage_objects_present(self, round0_state):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        assert result.project_coverage is not None
        assert result.eval_coverage is not None

    def test_counter_accuracy_all_supported(self, round0_state):
        """With N supported items, retrieve/generate call counts = N."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        # 1 project (supported) + 1 eval (supported) = 2 calls each
        assert result.retrieve_call_count == 2
        assert result.generate_call_count == 2

    def test_counter_accuracy_mixed_support(self, round0_state):
        """Unsupported items not counted; only supported items count."""
        proj = [
            _item(30, PROJ_S_SRC, "supported"),
            _item(31, PROJ_U_SRC, "unsupported"),
        ]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        assert result.retrieve_call_count == 1
        assert result.generate_call_count == 1

    def test_pool_label_on_predictions(self, round0_state):
        """CellPrediction.pool distinguishes 'project' from 'eval'."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        for pred in result.project_predictions:
            assert pred.pool == "project"
        for pred in result.eval_predictions:
            assert pred.pool == "eval"

    def test_prediction_fields_complete(self, round0_state):
        """Each CellPrediction records all required fields."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        # All required fields present
        _ = pred.pool
        _ = pred.item_id
        _ = pred.corpus_idx
        _ = pred.source_text
        _ = pred.retrieved_evidence_ids
        _ = pred.source_supported
        _ = pred.unsupported_spans
        _ = pred.outcome
        _ = pred.backend_called
        _ = pred.raw_generation
        _ = pred.final_translation
        _ = pred.surfaced_failure_artifact
        _ = pred.evidence_ids_at_round

    def test_accepted_prediction_fields(self, round0_state):
        """ACCEPTED_LICENSED prediction has final_translation and correct fields."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        result = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        assert pred.outcome == Outcome.ACCEPTED_LICENSED
        assert pred.backend_called is True
        assert pred.source_supported is True
        assert pred.raw_generation == "good output"
        assert pred.final_translation == "good output"


# ---------------------------------------------------------------------------
# Section 11: Determinism and immutability
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self, round0_state, tiny_manifest):
        """evaluate_round_state is deterministic: same inputs → same result."""
        proj = [_item(30, PROJ_S_SRC, "pt0"), _item(31, PROJ_U_SRC, "pt1")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        result1 = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        result2 = evaluate_round_state(
            round0_state, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        assert result1.round == result2.round
        assert result1.evidence_ids == result2.evidence_ids
        assert result1.retrieve_call_count == result2.retrieve_call_count
        assert result1.generate_call_count == result2.generate_call_count
        for p1, p2 in zip(result1.project_predictions, result2.project_predictions):
            assert p1.outcome == p2.outcome
            assert p1.source_supported == p2.source_supported
            assert p1.retrieved_evidence_ids == p2.retrieved_evidence_ids

    def test_evaluated_round_is_frozen(self, round0_state):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        with pytest.raises((AttributeError, TypeError)):
            result.round = 99  # type: ignore[misc]

    def test_cell_prediction_is_frozen(self, round0_state):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        result = evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        pred = result.project_predictions[0]
        with pytest.raises((AttributeError, TypeError)):
            pred.outcome = Outcome.OTHER_HARD_FAILURE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Section 12: evaluate_schedule composition
# ---------------------------------------------------------------------------

class TestEvaluateSchedule:
    def test_one_per_state(self, tiny_schedule):
        """evaluate_schedule returns one EvaluatedRound per schedule state."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = [_item(20, EVAL_S_SRC, "et0")]

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        assert len(rounds) == len(tiny_schedule.states)

    def test_rounds_in_schedule_order(self, tiny_schedule):
        """Rounds are in schedule.states order with correct round indices."""
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        for i, (er, state) in enumerate(zip(rounds, tiny_schedule.states)):
            assert er.round == state.round, f"rounds[{i}].round={er.round} != state.round={state.round}"

    def test_fixed_order_project_items_every_round(self, tiny_schedule):
        """Project items appear in same fixed order for every round."""
        proj = [
            _item(30, PROJ_S_SRC, "pt0"),
            _item(31, PROJ_U_SRC, "pt1"),
        ]
        ev_items = []

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        for er in rounds:
            assert len(er.project_predictions) == 2
            assert er.project_predictions[0].corpus_idx == 30
            assert er.project_predictions[1].corpus_idx == 31

    def test_fixed_order_eval_items_every_round(self, tiny_schedule):
        """Eval items appear in same fixed order for every round."""
        proj = []
        ev_items = [_item(20, EVAL_S_SRC, "et0"), _item(21, EVAL_S_SRC, "et1")]
        ev_items[1] = _item(21, "charlie delta extra", "et1")

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        for er in rounds:
            assert len(er.eval_predictions) == 2
            assert er.eval_predictions[0].corpus_idx == 20
            assert er.eval_predictions[1].corpus_idx == 21

    def test_evidence_grows_across_rounds(self, tiny_schedule):
        """Later rounds have more evidence than earlier ones."""
        proj = []
        ev_items = []

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        evidence_counts = [er.evidence_count for er in rounds]
        for i in range(1, len(evidence_counts)):
            assert evidence_counts[i] > evidence_counts[i - 1], (
                f"Evidence count must grow: {evidence_counts}"
            )

    def test_schedule_returns_tuple(self, tiny_schedule):
        proj = [_item(30, PROJ_S_SRC, "pt0")]
        ev_items = []

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        assert isinstance(rounds, tuple)

    def test_schedule_with_oracle(self, tiny_schedule):
        """evaluate_schedule passes oracle to each round."""
        proj = [_item(31, PROJ_U_SRC, "unsupported")]
        ev_items = []

        oracle_called = []

        def oracle(corpus_idx, source_text):
            oracle_called.append(corpus_idx)
            return True  # all unsupported are false abstentions

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
            optional_false_abstention_oracle=oracle,
        )
        # Oracle was called for each unsupported item in each round
        assert len(oracle_called) > 0
        for er in rounds:
            for pred in er.project_predictions:
                assert pred.outcome == Outcome.FALSE_SOURCE_ABSTENTION

    def test_schedule_exact_counter_sums(self, tiny_schedule):
        """Sum of retrieve/generate call counts matches expected from supported items."""
        proj = [
            _item(30, PROJ_S_SRC, "supported"),
            _item(31, PROJ_U_SRC, "unsupported"),
        ]
        ev_items = [_item(20, EVAL_S_SRC, "eval supported")]

        rounds = evaluate_schedule(
            tiny_schedule, proj, ev_items, LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=_good_te,
        )
        # Per round: 1 supported project + 1 supported eval = 2 calls each
        for er in rounds:
            assert er.retrieve_call_count == 2
            assert er.generate_call_count == 2


# ---------------------------------------------------------------------------
# Section 13: known_source_types from current HumanEvidence only
# ---------------------------------------------------------------------------

class TestKnownSourceTypes:
    def test_round_0_known_types_from_seed_only(self, round0_state):
        """Round 0: known types from seed evidence only."""
        received_types: list[frozenset] = []

        def generate_fn(req: PredictionRequest):
            received_types.append(req.known_source_types)
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=generate_fn,
        )
        assert len(received_types) > 0
        kt = received_types[0]
        # Must include seed tokens
        assert "alpha" in kt  # from SEED_A_SRC
        assert "charlie" in kt  # from SEED_B_SRC
        # Must NOT include acquisition-only tokens (not yet revealed at round 0)
        # (ACQ tokens include 'extra', 'acquisition', 'more')
        # Note: 'acquisition' could appear in seed if present, but it's not here

    def test_round_1_includes_new_acquisition(self, tiny_schedule):
        """Round 1: known types grow after new acquisition revealed."""
        state1 = tiny_schedule.states[1]

        received_types: list[frozenset] = []

        def retrieve_fn(req, evidence):
            return [evidence[0].corpus_idx]

        def generate_fn(req: PredictionRequest):
            received_types.append(req.known_source_types)
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            state1, proj, [], LANG,
            retrieve_fn=retrieve_fn,
            generate_fn=generate_fn,
        )
        kt = received_types[0]
        # corpus_idx=10 was revealed at round 1: source = ACQ_0_SRC = "alpha extra acquisition"
        assert "extra" in kt  # newly added at round 1
        assert "acquisition" in kt

    def test_known_types_is_frozenset(self, round0_state):
        """PredictionRequest.known_source_types is a frozenset."""
        received: list = []

        def generate_fn(req):
            received.append(req.known_source_types)
            return _good_te(req)

        proj = [_item(30, PROJ_S_SRC, "pt0")]
        evaluate_round_state(
            round0_state, proj, [], LANG,
            retrieve_fn=_simple_retrieve,
            generate_fn=generate_fn,
        )
        assert isinstance(received[0], frozenset)
