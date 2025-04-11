""" """

import os

os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-21-openjdk-amd64"
import time
import json
from cores.normalizer import TextNormalizer


TEST_TEXTS = [
    # "1. Những ngân hàng đang có lãi suất cho vay bình quân cao như Liên Việt, Bản Việt, Kiên Long với lãi suất từ 8,07 vnđ -  8,94VND...",
    # "2. Từ lâu công viên 30/4 đã trở thành nơi lý tưởng để giới trẻ, du khách chụp ảnh, nhâm nhi ly cà phê hay ngồi dưới tán cây trò chuyện tránh những ngày nắng nóng. Công viên 23-9 nằm giữa hai con đường Lê Lai và Phạm Ngũ Lão, trải dài từ công trường Quách Thị Trang đến đường Nguyễn Trãi. Công viên gồm 3 khu và bị ngăn cách nhau bởi đường Nguyễn Thị Nghĩa.",
    # "3. Bộ Quốc phòng Nga đồng thời tuyên bố hệ thống phòng không Nga đã bắn hạ 46 máy bay không người lái của Ukraine, một tên lửa ATACMS và 8 quả bom thông minh Hammer trong 24 giờ qua.",
    # "4. Lấy cảm hứng từ chiếc xe quan tài Dracula xuất hiện trong tập phim năm 1965 của series phim truyền hình The Munsters",
    # "5. Lãnh đạo Sở GD-ĐT TP.HCM cho biết về nguyên tắc xét điểm chuẩn lớp 10 trường, lớp chuyên như sau: Chỉ xét tuyển đối với thí sinh tham dự đủ các bài thi quy định, không vi phạm nội quy trong kỳ thi tuyển sinh và các bài thi đều đạt điểm lớn hơn 2.",
    # "Trở thành Software Engineer (SE) với mức lương Upto $3,500 tại VietDevelopers!",
    # "hôm nay là 23/03/2025.",
    # "Tôi có 5 con mèo. 100 người tham gia. Trận đấu diễn ra lúc 14:30. Cuộc họp bắt đầu lúc 9:05.",
    # "TP.HCM, ngày 30/04/2025 14:30. Tôi đổi 100 USD sang VNĐ. Giá xăng hôm nay: 22.500 VNĐ/lít. TP.HCM là thành phố lớn.",
    # "5kg gạo. Tôi đổi 100 USD sang VNĐ.",
    # "Theo WHO, số ca mắc COVID-19 đang có xu hướng giảm.",
    # "Giá xăng tăng lên 25.000đ/lít.",
    # "Sân bay mới có diện tích 120ha 1.000,50€ 50 - 100%.",
    "Phải nói thật nhé, thương mại điện tử giờ vừa tiện vừa... ức chế kinh khủng! Vui có, bực có, hào hứng có, mà đôi khi muốn \"xả stress\" cũng chỉ cần lướt app mua sắm vài vòng là đủ drama cho cả ngày.\nCó ai như mình không, vừa định săn sale iPhone 14 Pro Max 128GB, tưởng đâu sẽ là chiến lợi phẩm, thế mà trong vòng chưa đầy 1 giây — hết hàng! Cảm giác lúc đó như muốn \"bốc hoả\"! Thế nhưng chẳng lâu sau, mở qua thấy AirPods Pro (Giá: 5.200.000 VNĐ), Apple Watch Series 8 (Giá: 9.000.000 VNĐ), Smart TV LG 4K (Giá: 12.500.000 VNĐ), Robot hút bụi Xiaomi (Giá: 3.200.000 VNĐ) giảm giá sập sàn — cảm giác như trúng số vậy! Cảm xúc cứ như chơi tàu lượn siêu tốc, lên rồi xuống, chẳng thể nào bình tĩnh nổi.\nThế mà khi mua đồ gia dụng như Máy lạnh Daikin (Giá: 14.000.000 VNĐ), Panasonic (Giá: 10.500.000 VNĐ), LG (Giá: 11.000.000 VNĐ), Midea (Giá: 8.500.000 VNĐ), mình cứ nghĩ sẽ yên tâm với những món đồ xịn sò. Nhưng đọc review xong, chỉ muốn... khóc thét. Máy lạnh mát đấy nhưng mà ship về thì nguyên cái thùng móp méo như thể bị... đi phượt về. Trời ơi, ai mà không tức! Nhưng rồi, tìm được cái shop uy tín, giao hàng nhanh, bảo hành đầy đủ — tự dưng lại yêu thương mại điện tử như yêu người yêu cũ quay lại làm hòa vậy.\nMua đồ công nghệ thôi chưa đủ, giờ người ta còn tìm việc, tìm nhà, tìm phòng... online luôn. Mình đã thử apply CV qua các nền tảng như Viettel (Website: viettel.com.vn), DevUP (Email: hr@devup.com), và Freelancer — nhưng mà chờ hoài không thấy tin nhắn từ HR rep. Cảm giác lúc đó thì chỉ muốn hét lên: \"Ơ kìa, có đọc CV không mà cứ seen không trả lời!\" Nhưng mà, khi nhận được offer lương cao, job Remote, và Benefits/Commission hấp dẫn thì... ôi dào, bực mình lúc trước chợt biến mất.\nRiêng mảng bất động sản online cũng đầy cảm xúc dở khóc dở cười. Tìm thấy Vinhomes (Website: vinhomes.vn), Times City, Masteri, Sky Villa nghe sang chảnh vậy đó, nhưng khi bấm vô xem giá thì... muốn ngã ngửa: \"Ôi trời, làm sao mà mua được?\" Thế là đành quay lại tìm chung cư mini, giá hợp lý hơn, full nội thất, wifi mạnh, gần Vinmart, WC sạch sẽ — vậy cũng đủ vui rồi.\nCông nhận, thương mại điện tử chính là thế giới không ngủ. Một bên là người mua sẵn sàng \"chốt đơn thần tốc\", một bên là người bán canh deal, tối ưu ads, chạy KPI mệt mỏi. Các thương hiệu thì cứ liên tục đua nhau tung ra sản phẩm mới, với đủ loại Marketing, Livestream, Digital đến mức người dùng chỉ biết thốt lên: \"Trời ơi, ví em không chịu nổi nữa đâu!\" Và đó... là cái hay, cái điên, cái vui, cái tức... mà chỉ có ai mê thương mại điện tử mới hiểu được hết!"
]


if __name__ == "__main__":
    tik = time.time()
    text_normalizer = TextNormalizer("./exps/vncorenlp/")
    print(f"[*] take {time.time() - tik} seconds to load model")
    tik = time.time()
    for text_inputs in TEST_TEXTS:
        text_outputs = text_normalizer(text_inputs)
        print (text_outputs)
    print(f"[*] take {time.time() - tik} seconds to normalize {len(TEST_TEXTS)} texts")
