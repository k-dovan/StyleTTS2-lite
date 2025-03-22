from utils.text_processing import convert_time_to_words

def test_convert_time_to_words():
    assert convert_time_to_words("Trận đấu diễn ra lúc 14:30.") == "Trận đấu diễn ra lúc mười bốn giờ ba mươi phút."
    assert convert_time_to_words("Cuộc họp bắt đầu lúc 9:05.") == "Cuộc họp bắt đầu lúc chín giờ năm phút."
