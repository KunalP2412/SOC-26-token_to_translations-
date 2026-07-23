# from basic import BasicTokenizer

# tok = BasicTokenizer()
# tok.train("hello hello hello world world", vocab_size=260, verbose=True)

# ids = tok.encode("hello world")
# print(ids)
# print(tok.decode(ids))


from regex_tokenizer import RegexTokenizer

tok = RegexTokenizer()
tok.train("hello hello hello world world", vocab_size=260, verbose=True)
print(tok.encode("hello world"))