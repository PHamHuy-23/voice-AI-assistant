"""Correct the model name and architecture description in the normalized report."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "reports" / "midterm" / (
    "Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi_cong_thuc_da_chinh.docx"
)
OUTPUT = SOURCE.with_name(
    "Bao_cao_hang_muc_cham_giua_ky_Xu_ly_tieng_noi_"
    "cong_thuc_va_ten_mo_hinh_da_chinh.docx"
)


def set_paragraph_text(paragraph, text: str) -> None:
    """Replace text while retaining the formatting of the first run."""
    if paragraph.runs:
        paragraph.runs[0].text = text
        for run in paragraph.runs[1:]:
            run.text = ""
    else:
        paragraph.add_run(text)


def replace_in_runs(paragraph, replacements: list[tuple[str, str]]) -> int:
    count = 0
    for run in paragraph.runs:
        original_value = run.text
        value = original_value
        for old, new in replacements:
            if old in value:
                occurrences = value.count(old)
                value = value.replace(old, new)
                count += occurrences
        # Assigning run.text rebuilds the run contents and removes non-text
        # children such as inline drawings. Only touch runs whose visible text
        # actually changed so image-only runs remain intact.
        if value != original_value:
            run.text = value
    return count


CORRECTED_ENCODER_CODE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        padding = (3 * dilation, 0)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(7, 1),
            stride=1, padding=padding,
            dilation=(dilation, 1), bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(7, 1),
            stride=1, padding=padding,
            dilation=(dilation, 1), bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        residual = F.relu(self.shortcut(x))
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class TCResNet8Dilated(nn.Module):
    """Tên implementation trên Kaggle của encoder TD-ResNet7."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            40, 16, kernel_size=(3, 1),
            stride=1, padding=(1, 0), bias=False
        )
        self.res1 = DilatedResidualBlock(16, 24, dilation=1)
        self.res2 = DilatedResidualBlock(24, 32, dilation=2)
        self.res3 = DilatedResidualBlock(32, 48, dilation=4)
        self.avg_pool = nn.AvgPool2d((51, 1))

    def forward(self, x):
        # [B, 1, 51, 40] -> [B, 40, 51, 1]
        x = x.squeeze(1).transpose(1, 2).unsqueeze(-1)
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.avg_pool(x).flatten(1)  # z in R^48'''


EXACT_REPLACEMENTS = {
    "Giải pháp được lựa chọn cho đồ án là mô hình Few-Shot Keyword Spotting sử dụng kiến trúc TC-ResNet kết hợp Prototypical Network (ProtoNet).":
        "Giải pháp được lựa chọn cho đồ án là mô hình Few-Shot Keyword Spotting sử dụng kiến trúc TD-ResNet7 kết hợp Prototypical Network (ProtoNet).",
    "Trong một số tài liệu tổng quan ban đầu, mô hình trích xuất đặc trưng có thể được gọi là TD-ResNet (Time-Delay ResNet). Tuy nhiên, trong mã nguồn triển khai chính thức của repository Few-Shot-KWS mà nhóm kế thừa và tái hiện, module mạng được định nghĩa tường minh dưới lớp TCResNet (Temporal Convolutional ResNet). Bản chất của kiến trúc này là sử dụng các khối tích chập 1 chiều (1D Convolution) với kích thước kernel 9 × 1 quét dọc theo trục thời gian (Time-axis), dựa trên nền tảng của mạng nơ-ron tích chập thời gian (Temporal Convolutional Network). Để đảm bảo tính chính xác, khoa học và trung thực tuyệt đối với mã nguồn thực nghiệm, báo cáo sử dụng thống nhất danh xưng kỹ thuật TC-ResNet cho toàn bộ hệ thống.":
        "Paper Few-Shot Keyword Spotting With Prototypical Networks đề xuất TD-ResNet7 bằng cách sửa đổi kiến trúc nền TC-ResNet8: giảm kernel trong các khối phần dư xuống 7 × 1, giữ stride bằng 1 và áp dụng dilation lần lượt 1, 2, 4. Trong pipeline Kaggle, encoder này được đăng ký bằng tên implementation TCResNet8Dilated. Vì vậy, báo cáo sử dụng tên khoa học TD-ResNet7; tên TCResNet8Dilated chỉ được giữ nguyên khi trích dẫn class, tham số dòng lệnh hoặc checkpoint của mã nguồn thực nghiệm.",
    "3.1.2. TC-ResNet đóng vai trò embedding network":
        "3.1.2. TD-ResNet7 đóng vai trò embedding network",
    "Khác với các kiến trúc nhận dạng tiếng nói truyền thống sử dụng một lớp Fully Connected và Softmax ở tầng cuối cùng để phân loại một tập nhãn cố định, trong giải pháp Few-Shot Keyword Spotting, TC-ResNet đóng vai trò là một mạng mã hóa embedding (Feature Encoder):":
        "Khác với các kiến trúc nhận dạng tiếng nói truyền thống sử dụng một lớp Fully Connected và Softmax ở tầng cuối cùng để phân loại một tập nhãn cố định, trong giải pháp Few-Shot Keyword Spotting, TD-ResNet7 đóng vai trò là mạng mã hóa embedding (Feature Encoder):",
    "-- f_\\theta: Mạng nơ-ron tích chập thời gian TC-ResNet với tập tham số trọng số θ.":
        "-- f_\\theta: Mạng nơ-ron tích chập thời gian TD-ResNet7 với tập tham số trọng số θ.",
    "TC-ResNet không trực tiếp dự đoán tên từ khóa. Mục tiêu huấn luyện của mạng là ánh xạ các tín hiệu âm thanh có cùng bản chất ngữ âm (phonetic features) vào các vị trí lân cận nhau trong không gian vector, tạo tiền đề cho quá trình đối chiếu khoảng cách metric.":
        "TD-ResNet7 không trực tiếp dự đoán tên từ khóa. Mục tiêu huấn luyện của mạng là ánh xạ các tín hiệu âm thanh có cùng bản chất ngữ âm (phonetic features) vào các vị trí lân cận nhau trong không gian vector, tạo tiền đề cho quá trình đối chiếu khoảng cách metric.",
    "3.1.4. Kiến trúc TCResNet8": "3.1.4. Kiến trúc TD-ResNet7",
    "Mô hình TC-ResNet phiên bản 8 lớp (TCResNet8) được triển khai trong repository với cấu hình tham số kỹ thuật chuẩn:":
        "Encoder được huấn luyện trên Kaggle là TD-ResNet7 của paper, được đăng ký trong mã nguồn bằng tên TCResNet8Dilated, với cấu hình kỹ thuật:",
    "-- Các lớp Conv trong Residual blocks: Sử dụng kernel kích thước 9 × 1, quét dài theo trục thời gian để nắm bắt ngữ cảnh âm thanh rộng.":
        "-- Các lớp Conv trong Residual blocks: Sử dụng kernel kích thước 7 × 1 theo đúng kiến trúc TD-ResNet7 của paper.",
    "-- Lớp Conv khởi tạo đầu tiên (conv1): Sử dụng kernel kích thước 3 × 1, đưa số kênh từ 1 lên 16.":
        "-- Lớp Conv khởi tạo đầu tiên (conv1): Sau phép Time-Channel transformation, 40 hệ số MFCC trở thành 40 kênh đầu vào; lớp Conv kernel 3 × 1 ánh xạ 40 kênh này thành 16 kênh đặc trưng.",
    "-- Dilation: Được đặt cố định [1, 1, 1, 1] trong cấu hình chuẩn của TCResNet8.":
        "-- Dilation và stride: Ba khối phần dư sử dụng dilation [1, 2, 4] và stride bằng 1.",
    "Sơ đồ luồng xử lý bên trong mạng TCResNet8:":
        "Sơ đồ luồng xử lý bên trong mạng TD-ResNet7:",
    "3.3.4. Kiến trúc mạng mã hóa TC-ResNet8 Dilated":
        "3.3.4. Kiến trúc mạng mã hóa TD-ResNet7",
    "Mạng mã hóa TC-ResNet8 Dilated được xây dựng với 3 khối Residual Blocks, sử dụng tích chập thời gian 1D (kernel 9×1) và hệ số giãn nở tăng dần [1, 2, 4] nhằm nắm bắt ngữ cảnh âm vị dài hạn với chi phí tham số siêu nhỏ gọn (chỉ 64.560 tham số):":
        "Mạng mã hóa TD-ResNet7 được xây dựng với 3 khối Residual Blocks, sử dụng tích chập thời gian 1D với kernel 7 × 1, stride bằng 1 và hệ số giãn nở tăng dần [1, 2, 4]. Trong mã chạy Kaggle, kiến trúc này được đăng ký bằng tên TCResNet8Dilated. Mô hình tạo embedding 48 chiều và có 64.560 tham số:",
    "A. Những điểm tương đồng và xác thực thành công: Cả nghiên cứu gốc và dữ liệu tái hiện đều ghi nhận xu hướng tăng vọt về độ chính xác từ 1-shot lên 5-shot, sau đó bắt đầu bão hòa từ 10-shot đến 20-shot. Mốc 2-way 1-shot đạt 85.40% và 2-way 5-shot đạt 92.67% khớp hoàn hảo với các mốc công bố. Mức trần 95.40% tại 15-shot khẳng định mạng TC-ResNet8 Dilated tái hiện chính xác 100% năng lực biểu diễn mêtric của tác giả.":
        "A. Những điểm tương đồng và kết quả tái hiện: Cả nghiên cứu gốc và dữ liệu tái hiện đều ghi nhận xu hướng tăng độ chính xác rõ rệt từ 1-shot lên 5-shot, sau đó dần bão hòa từ 10-shot đến 20-shot. Mốc 2-way 1-shot đạt 85,40% và 2-way 5-shot đạt 92,67%, phù hợp với các mốc công bố. Mức cao nhất 95,40% tại 15-shot cho thấy encoder TD-ResNet7 kết hợp Prototypical Network tái hiện tốt xu hướng và năng lực biểu diễn metric của phương pháp gốc.",
}


GENERIC_REPLACEMENTS = [
    ("Kernel 9×1", "Kernel 7×1"),
    ("kernel 9×1", "kernel 7×1"),
    ("TD-RESNET", "TD-RESNET7"),
    ("TD-ResNet", "TD-ResNet7"),
    ("TC-ResNet8 Dilated", "TD-ResNet7"),
    ("TC-ResNet8", "TD-ResNet7"),
    ("TCResNet8", "TD-ResNet7"),
    ("TC-RESNET8", "TD-RESNET7"),
    ("TC-ResNet", "TD-ResNet7"),
]


def is_code_paragraph(paragraph) -> bool:
    text = paragraph.text
    return "\n" in text and any(
        marker in text
        for marker in ("class ", "import ", "--model.encoding=", "def ", "torch.")
    )


def main() -> None:
    doc = Document(SOURCE)
    exact_count = 0
    generic_count = 0

    for paragraph in doc.paragraphs:
        original = paragraph.text
        if original in EXACT_REPLACEMENTS:
            set_paragraph_text(paragraph, EXACT_REPLACEMENTS[original])
            exact_count += 1
            continue
        if "class TCResNet8Dilated" in original and "kernel_size=(9, 1)" in original:
            set_paragraph_text(paragraph, CORRECTED_ENCODER_CODE)
            exact_count += 1
            continue
        if is_code_paragraph(paragraph):
            # Update prose-only docstrings without changing executable identifiers.
            generic_count += replace_in_runs(
                paragraph,
                [("mạng TC-ResNet.", "mạng TD-ResNet7.")],
            )
            continue
        generic_count += replace_in_runs(paragraph, GENERIC_REPLACEMENTS)

    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    # Keep literal CLI/class identifiers, but explain them in tables.
                    if paragraph.text.strip() == "TCResNet8Dilated":
                        set_paragraph_text(
                            paragraph,
                            "TCResNet8Dilated (tên implementation của TD-ResNet7)",
                        )
                        exact_count += 1
                    elif "--model.encoding=TCResNet8Dilated" not in paragraph.text:
                        generic_count += replace_in_runs(paragraph, GENERIC_REPLACEMENTS)

    # A few table cells contain prose snapshots from the demo output.
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    replace_in_runs(
                        paragraph,
                        [("Official TD-ResNet7", "TD-ResNet7 theo paper")],
                    )

    doc.save(OUTPUT)

    # Text boxes used by the architecture diagram are not exposed through
    # python-docx's paragraph collection. Patch only the stale kernel labels
    # in the underlying Word XML while leaving executable identifiers intact.
    with ZipFile(OUTPUT, "r") as source_zip, NamedTemporaryFile(
        suffix=".docx", delete=False, dir=OUTPUT.parent
    ) as temp_file:
        temp_path = Path(temp_file.name)
        with ZipFile(temp_file, "w", ZIP_DEFLATED) as target_zip:
            for item in source_zip.infolist():
                data = source_zip.read(item.filename)
                if item.filename.startswith("word/") and item.filename.endswith(".xml"):
                    data = data.replace(b"Kernel 9x1", b"Kernel 7x1")
                    data = data.replace(b"kernel 9x1", b"kernel 7x1")
                target_zip.writestr(item, data)
    temp_path.replace(OUTPUT)

    print(f"Saved: {OUTPUT}")
    print(f"Exact corrections: {exact_count}")
    print(f"Terminology replacements: {generic_count}")


if __name__ == "__main__":
    main()
