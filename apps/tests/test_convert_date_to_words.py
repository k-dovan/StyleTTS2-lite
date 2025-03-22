from utils.text_processing import convert_date_to_words

def test_convert_date_to_words():
    assert convert_date_to_words("hôm nay là 23/03/2025.") == "hôm nay là ngày hai mươi ba tháng ba năm hai nghìn không trăm hai mươi lăm."
    assert convert_date_to_words("ngày 2025-04-30 là lễ lớn.") == "ngày ba mươi tháng tư năm hai nghìn không trăm hai mươi lăm là lễ lớn."
