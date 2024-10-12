# Based on seminar materials

# Don't forget to support cases when target_text == ''

def lev_dist(seq1, seq2):
    len_seq1, len_seq2 = len(seq1), len(seq2)
    dp = [[0] * (len_seq2 + 1) for _ in range(len_seq1 + 1)]

    for i in range(len_seq1 + 1):
        dp[i][0] = i
    for j in range(len_seq2 + 1):
        dp[0][j] = j

    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[len_seq1][len_seq2]


def calc_cer(target_text, predicted_text) -> float:
    # TODO
    target_text = target_text.replace(" ", "")
    predicted_text = predicted_text.replace(" ", "")

    if not target_text:
        return 1. if predicted_text else 0.
    
    cer = lev_dist(target_text, predicted_text) / max(len(target_text), 1)
    return cer


def calc_wer(target_text, predicted_text) -> float:
    # TODO
    target_words = target_text.split()
    predicted_words = predicted_text.split()

    if not target_words:
        return 1. if predicted_words else 0.

    wer = lev_dist(target_words, predicted_words) / len(target_words)
    return wer