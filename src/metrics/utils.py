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
                cost = 0  
            else:
                cost = 1 

            dp[i][j] = min(
                dp[i - 1][j] + 1,        
                dp[i][j - 1] + 1,        
                dp[i - 1][j - 1] + cost  
            )

    return dp[len_seq1][len_seq2]


def calc_cer(target_text, predicted_text) -> float:
    target_text = target_text.replace(" ", "")
    predicted_text = predicted_text.replace(" ", "")

    if not target_text:
        return 1.0

    cer = lev_dist(target_text, predicted_text) / len(target_text)
    return cer


def calc_wer(target_text, predicted_text) -> float:
    target_words = target_text.strip().split()
    predicted_words = predicted_text.strip().split()

    if not target_words:
        return 1.0

    wer = lev_dist(target_words, predicted_words) / len(target_words)
    return wer

