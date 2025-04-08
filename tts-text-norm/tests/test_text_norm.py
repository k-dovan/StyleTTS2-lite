import os
import pytest

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
from cores.normalizer import TextNormalizer

text_normalizer = TextNormalizer("./exps/vncorenlp/")

# 🔠 Abbreviation Expansion
@pytest.mark.parametrize("input_text, expected_output", [
    ("TP.HCM đang đối mặt với tình trạng kẹt xe nghiêm trọng.",
     "thành phố hồ chí minh đang đối mặt với tình trạng kẹt xe nghiêm trọng ."),
    ("Bộ GD&ĐT vừa ban hành quy chế thi tốt nghiệp THPT năm 2025.",
     "bộ giáo dục và đào tạo vừa ban hành quy chế thi tốt nghiệp trung học phổ thông năm hai nghìn không trăm hai mươi lăm ."),
    ("Theo WHO, số ca mắc COVID-19 đang có xu hướng giảm.",
     "theo vê kép hát ô, số ca mắc cô vít mười chín đang có xu hướng giảm ."),
])
def test_abbreviation_expansion(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 📅 Date Normalization
@pytest.mark.parametrize("input_text, expected_output", [
    ("Hội nghị diễn ra vào ngày 15/04/2025.",
     "hội nghị diễn ra vào ngày mười lăm tháng tư năm hai nghìn không trăm hai mươi lăm ."),
    ("Bài viết được đăng ngày 8-3.",
     "bài viết được đăng ngày tám tháng ba ."),
    ("Tết Nguyên đán rơi vào ngày 29.01.2025.",
     "tết nguyên đán rơi vào ngày hai mươi chín tháng một năm hai nghìn không trăm hai mươi lăm ."),
])
def test_date_normalization(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# ⏰ Time Expressions
@pytest.mark.parametrize("input_text, expected_output", [
    ("Trận đấu bắt đầu lúc 19h30.",
     "trận đấu bắt đầu lúc mười chín giờ ba mươi ."),
    ("Hội thảo diễn ra vào 8 giờ sáng mai.",
     "hội thảo diễn ra vào tám giờ sáng mai ."),
    ("Chuyến bay cất cánh lúc 22:45.",
     "chuyến bay cất cánh lúc hai mươi hai giờ bốn mươi lăm phút ."),
])
def test_time_expressions(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 💰 Currency & Numbers
@pytest.mark.parametrize("input_text, expected_output", [
    ("Giá xăng tăng lên 25.000đ/lít.",
     "giá xăng tăng lên hai mươi lăm nghìn đồng trên lít ."),
    ("Mức lương tối thiểu là 4,680,000 VNĐ.",
     "mức lương tối thiểu là bốn triệu sáu trăm tám mươi nghìn đồng ."),
    ("Gói cứu trợ trị giá 2 tỷ đồng.",
     "gói cứu trợ trị giá hai tỷ đồng ."),
])
def test_currency_and_numbers(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 📏 Units & Measurements
@pytest.mark.parametrize("input_text, expected_output", [
    ("Sân bay mới có diện tích 120ha.",
     "sân bay mới có diện tích một trăm hai mươi hecta ."),
    ("Trẻ em nên uống 500ml sữa mỗi ngày.",
     "trẻ em nên uống năm trăm mi li lít sữa mỗi ngày ."),
    ("Anh ấy cao 1m75.",
     "anh ấy cao một mét bảy mươi lăm ."),
])
def test_units_and_measurements(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🔢 General Numbers
@pytest.mark.parametrize("input_text, expected_output", [
    ("Dân số Việt Nam năm 2025 dự kiến đạt 105 triệu người.",
     "dân số việt nam năm hai nghìn không trăm hai mươi lăm dự kiến đạt một trăm lẻ năm triệu người ."),
    ("Tỷ lệ thất nghiệp giảm còn 3.5%.",
     "tỷ lệ thất nghiệp giảm còn ba phẩy năm phần trăm ."),
    ("Có hơn 12.345 ca mắc mới được ghi nhận.",
     "có hơn mười hai nghìn ba trăm bốn mươi lăm ca mắc mới được ghi nhận ."),
])
def test_general_numbers(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🧾 Common News Phrases
@pytest.mark.parametrize("input_text, expected_output", [
    ("Theo ghi nhận của PV, tình trạng ngập kéo dài.",
     "theo ghi nhận của phóng viên, tình trạng ngập kéo dài ."),
    ("Chính phủ sẽ họp vào đầu tuần tới để đưa ra quyết định.",
     "chính phủ sẽ họp vào đầu tuần tới để đưa ra quyết định ."),
    ("Thủ tướng phát biểu tại lễ khai mạc diễn ra ở Hà Nội.",
     "thủ tướng phát biểu tại lễ khai mạc diễn ra ở hà nội ."),
])
def test_common_phrases(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🌐 Websites & Links
@pytest.mark.parametrize("input_text, expected_output", [
    ("Độc giả có thể xem thêm tại www.vietnamnews.vn.",
     "độc giả có thể xem thêm tại www chấm vietnamnews chấm v n ."),
    ("Tham khảo thêm thông tin qua địa chỉ https://moet.gov.vn.",
     "tham khảo thêm thông tin qua địa chỉ h t t p s hai chấm gạch gạch moet chấm g o v chấm v n ."),
])
def test_web_links(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🧠 Special Characters & Slang
@pytest.mark.parametrize("input_text, expected_output", [
    ("Tin nóng!!!", "tin nóng ."),
    ("Khẩn cấp: Cảnh báo mưa lớn.", "khẩn cấp: cảnh báo mưa lớn ."),
    ("Giá điện tăng “chóng mặt”.", "giá điện tăng chóng mặt ."),
])
def test_special_characters(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🇻🇳 Vietnamese Names & Locations
@pytest.mark.parametrize("input_text, expected_output", [
    ("Ông Nguyễn Văn A đã phát biểu tại buổi lễ.",
     "ông nguyễn văn a đã phát biểu tại buổi lễ ."),
    ("Cầu Cần Thơ là công trình tiêu biểu ở miền Tây.",
     "cầu cần thơ là công trình tiêu biểu ở miền tây ."),
])
def test_names_and_locations(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🧪 Mixed Context & Edge Cases
@pytest.mark.parametrize("input_text, expected_output", [
    ("Thời tiết hôm nay tại TP.HCM: 32°C, độ ẩm 75%.",
     "thời tiết hôm nay tại thành phố hồ chí minh: ba mươi hai độ c, độ ẩm bảy mươi lăm phần trăm ."),
    ("Ông ấy nặng 75kg và cao 1m80.",
     "ông ấy nặng bảy mươi lăm ki lô gam và cao một mét tám mươi ."),
])
def test_mixed_edge_cases(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output

# 🧪 Multi-Sentence Test Cases
@pytest.mark.parametrize("input_text, expected_output", [
    ("TP.HCM đang mưa lớn. Người dân nên hạn chế ra đường.",
     "thành phố hồ chí minh đang mưa lớn . người dân nên hạn chế ra đường ."),
    ("Hội nghị sẽ diễn ra vào ngày 20/4. Thời gian bắt đầu lúc 9h sáng.",
     "hội nghị sẽ diễn ra vào ngày hai mươi tháng tư . thời gian bắt đầu lúc chín giờ sáng ."),
    ("Giá vàng hôm nay tăng mạnh. Mức giá hiện tại là 75.000.000đ/lượng.",
     "giá vàng hôm nay tăng mạnh . mức giá hiện tại là bảy mươi lăm triệu đồng một lượng ."),
    ("Website www.tuoitre.vn cập nhật tin nhanh. Mời độc giả đón đọc.",
     "website www chấm tuoitre chấm v n cập nhật tin nhanh . mời độc giả đón đọc ."),
    ("Trận đấu bắt đầu lúc 19h30. Dự kiến kết thúc lúc 21h45.",
     "trận đấu bắt đầu lúc mười chín giờ ba mươi phút . dự kiến kết thúc lúc hai mươi một giờ bốn mươi lăm phút ."),
    ("Bé nặng 3.5kg khi sinh. Hiện đã được 6 tháng tuổi.",
     "bé nặng ba phẩy năm ki lô gam khi sinh . hiện đã được sáu tháng tuổi ."),
    ("Theo báo cáo, tỷ lệ thất nghiệp là 3.7%. Con số này giảm nhẹ so với quý trước.",
     "theo báo cáo , tỷ lệ thất nghiệp là ba phẩy bảy phần trăm . con số này giảm nhẹ so với quý trước ."),
    ("Bộ GD&ĐT vừa công bố lịch thi. Kỳ thi sẽ bắt đầu từ ngày 1/6.",
     "bộ giáo dục và đào tạo vừa công bố lịch thi . kỳ thi sẽ bắt đầu từ ngày một tháng sáu ."),
    ("Anh ấy cao 1m80. Cân nặng 80kg.",
     "anh ấy cao một mét tám mươi . cân nặng tám mươi ki lô gam ."),
    ("Nhiệt độ ở Hà Nội là 32°C. Trong khi đó, TP.HCM là 34°C.",
     "nhiệt độ ở hà nội là ba mươi hai độ c . trong khi đó , thành phố hồ chí minh là ba mươi bốn độ c ."),
])
def test_multi_sentence_normalization(input_text, expected_output):
    assert ' '.join(text_normalizer(input_text)) == expected_output
