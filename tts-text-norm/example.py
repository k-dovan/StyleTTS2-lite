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
    "Website www.tuoitre.vn cập nhật tin nhanh. Mời độc giả đón đọc."
]


if __name__ == "__main__":
    tik = time.time()
    text_normalizer = TextNormalizer("./exps/vncorenlp/")
    print(f"[*] take {time.time() - tik} seconds to load model")
    tik = time.time()
    for text_inputs in TEST_TEXTS:
        text_outputs = text_normalizer(text_inputs)
        print (' '.join(text_outputs))
    print(f"[*] take {time.time() - tik} seconds to normalize {len(TEST_TEXTS)} texts")
