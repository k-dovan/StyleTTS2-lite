from utils.text_processing import expand_abbreviations

def test_expand_abbreviations():
    assert expand_abbreviations("TP.HCM là thành phố lớn.") == "Thành phố Hồ Chí Minh là thành phố lớn."
    assert expand_abbreviations("5kg gạo") == "5 kilogram gạo"
    assert expand_abbreviations("10km đường") == "10 kilômét đường"
    assert expand_abbreviations("Tôi đổi 100 USD sang VNĐ.") == "Tôi đổi 100 đô la Mỹ sang Việt Nam đồng."
