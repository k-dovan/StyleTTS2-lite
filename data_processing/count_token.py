import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o-mini")
text = "bình thường mọi người khi mà nhìn thấy dọc cái bờ hồ này sẽ"
tokens = enc.encode(text)
print(len(tokens))  # This prints the exact token count
