from hsd.preprocess import TextCleaner, clean_text


def test_clean_lowercases_and_strips_urls():
    out = clean_text("Check THIS http://spam.example.com out", lowercase=True, strip_urls=True)
    assert "http" not in out
    assert out == "check this out"


def test_clean_removes_mentions():
    out = clean_text("hey @user how are you", strip_mentions=True)
    assert "@user" not in out
    assert "how are you" in out


def test_clean_collapses_repeats():
    out = clean_text("soooooo good", collapse_repeats=True)
    assert "soo good" == out


def test_hashtag_keeps_word_by_default():
    out = clean_text("this is #awesome", strip_hashtags=False)
    assert "awesome" in out
    assert "#" not in out


def test_cleaner_transformer_handles_non_strings():
    cleaner = TextCleaner()
    out = cleaner.transform(["Hello WORLD", None, 123])
    assert out[0] == "hello world"
    assert out[1] == ""
    assert out[2] == "123"
