from evaluation.metrics import citation_coverage, cited_claim_count


def test_citation_metrics_detect_markers() -> None:
    answer = "Transformers improved AUROC [abc-123]. CNNs were competitive [def-456]."

    cov = citation_coverage(answer, ["abc-123", "def-456"])
    claims = cited_claim_count(answer)

    assert cov.value == 1.0
    assert claims.value == 2.0
