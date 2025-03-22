from utils.text_processing import convert_number_to_words

def test_convert_number_to_words():
    assert convert_number_to_words("Tôi có 5 con mèo.") == "Tôi có năm con mèo."
    assert convert_number_to_words("100 người tham gia.") == "một trăm người tham gia."
