from utils.text_processing import process_text_for_tts

def test_process_text_for_tts():
    assert process_text_for_tts("TP.HCM, ngày 30/04/2025 14:30") == "Thành phố Hồ Chí Minh, ngày ba mươi tháng tư năm hai nghìn không trăm hai mươi lăm lúc mười bốn giờ ba mươi phút"
    assert process_text_for_tts("Tôi đổi 100 USD sang VNĐ.") == "Tôi đổi một trăm đô la Mỹ sang Việt Nam đồng."
    assert process_text_for_tts("Giá xăng hôm nay: 22.500 VNĐ/lít.") == "Giá xăng hôm nay: hai mươi hai nghìn năm trăm Việt Nam đồng một lít."
    assert process_text_for_tts("5kg gạo giá 10.000 VNĐ.") == "năm kilogram gạo giá mười nghìn Việt Nam đồng."
