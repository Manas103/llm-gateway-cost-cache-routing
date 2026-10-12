from llmgw.routing import classify_difficulty, generate_tasks


def test_generate_tasks_is_deterministic_for_a_fixed_seed():
    a = generate_tasks(50, seed=3)
    b = generate_tasks(50, seed=3)
    assert a == b


def test_generate_tasks_covers_all_three_difficulties():
    tasks = generate_tasks(300, seed=20260801)
    difficulties = {t.true_difficulty for t in tasks}
    assert difficulties == {"easy", "medium", "hard"}


def test_classify_difficulty_respects_thresholds():
    assert classify_difficulty(0.5, (1.5, 2.5)) == "easy"
    assert classify_difficulty(2.0, (1.5, 2.5)) == "medium"
    assert classify_difficulty(3.0, (1.5, 2.5)) == "hard"


def test_proxy_score_is_correlated_with_true_difficulty_but_not_identical():
    tasks = generate_tasks(300, seed=20260801)
    easy_scores = [t.proxy_score for t in tasks if t.true_difficulty == "easy"]
    hard_scores = [t.proxy_score for t in tasks if t.true_difficulty == "hard"]
    assert sum(easy_scores) / len(easy_scores) < sum(hard_scores) / len(hard_scores)
    # noisy, not perfectly separable: at least one "easy" task scores above at least one "hard" task
    assert max(easy_scores) > min(hard_scores)
