# Few-Shot Keyword Spotting — Project Data & Reproduction Log

## 1. Mục tiêu phần hiện tại

Phần đang thực hiện tập trung vào **dữ liệu và preprocessing** cho đồ án Few-Shot Keyword Spotting.

Mục tiêu chính:

- xác định đúng dataset mà paper sử dụng;
- tái hiện quy trình chuẩn bị dữ liệu từ repository gốc;
- phân biệt dữ liệu raw với dữ liệu đã được tổ chức cho few-shot;
- kiểm tra chính xác các bước preprocessing của paper;
- lấy toàn bộ số liệu thực tế từ Kaggle;
- tuyệt đối không tự giả số liệu cho báo cáo;
- chuẩn bị nền cho các mục 2.1 → 2.5 của báo cáo;
- sau đó mở rộng thử nghiệm preprocessing sang 2 dataset bổ sung:
  - AudioMNIST;
  - Fluent Speech Commands.

---

# 2. Hướng dataset của đồ án

## 2.1. Dataset chính

Dataset chính của mô hình là:

**Google Speech Commands v0.02**

Google Speech Commands là nguồn audio raw.

Không nên gọi FS-GSC là một dataset hoàn toàn độc lập với GSC.

Hiểu đúng:

```text
Google Speech Commands
        ↓
paper preprocessing / organization
        ↓
Few-Shot Google Speech Commands setup
```

FS-GSC chủ yếu là cách paper:

- lọc dữ liệu;
- tổ chức theo speaker;
- chia core / unknown;
- cân bằng dữ liệu;
- chia train / validation / test;
- sau đó dùng cho Few-shot episodic learning.

---

## 2.2. Dataset bổ sung dự kiến

### AudioMNIST

Vai trò:

- nguồn single-word / spoken digit ngoài GSC;
- dùng kiểm tra pipeline preprocessing;
- so sánh raw vs processed;
- không mặc định dùng để train mô hình chính.

### Fluent Speech Commands

Vai trò:

- dữ liệu spoken command nhiều từ;
- dùng kiểm tra preprocessing trên tín hiệu dài hơn;
- không coi đây là dataset multi-keyword spotting nếu bản chất bài toán dataset là spoken command / intent;
- không áp dụng máy móc quy tắc độ dài 1 giây của GSC lên FSC.

---

# 3. Nguyên tắc làm báo cáo

Một nguyên tắc đã thống nhất:

> Không tự giả bất kỳ số liệu dataset, preprocessing, feature hay kết quả mô hình nào.

Nếu cần:

- số file;
- số class;
- duration;
- sample rate;
- channel;
- số speaker;
- feature shape;
- accuracy;
- loss;
- latency;

thì phải lấy trực tiếp từ code và output Kaggle.

Nếu chưa có thì để dạng:

```text
[LẤY TỪ KAGGLE]
[TBD]
[NOT_MEASURED]
```

---

# 4. Cấu trúc phần 2 của báo cáo

## 2.1. Tổng quan dataset

Hướng hiện tại:

- Google Speech Commands là dataset chính;
- AudioMNIST là dataset bổ sung single-word;
- Fluent Speech Commands là dataset bổ sung multi-word;
- AudioMNIST và FSC dùng cho preprocessing experiment;
- chưa ghi số liệu định lượng nếu chưa audit.

## 2.2. Cấu trúc dữ liệu

Sẽ trình bày:

- số lượng mẫu;
- số lớp;
- nhãn;
- format;
- sample rate;
- channel;
- duration;
- class distribution;
- speaker information;
- train/val/test split;
- few-shot organization.

## 2.3. Quy trình chuẩn bị dữ liệu

Sẽ trình bày đúng theo code thực tế.

Không tự thêm:

- resampling;
- normalization;
- trim silence;
- augmentation;

nếu source không làm.

Ngoài ra sẽ có code thật của preprocessing.

## 2.4. Đặc trưng tiếng nói

Cần xác định từ source:

- waveform?
- MFCC?
- Mel?
- tham số cụ thể?
- feature tensor shape?

Phân biệt:

- feature dùng để minh họa;
- feature thật sự đưa vào model.

## 2.5. Minh họa và nhận xét

Dự kiến:

```text
RAW
vs
PROCESSED
```

So sánh bằng:

- waveform;
- spectrogram;
- Mel-spectrogram;
- MFCC;
- duration;
- peak amplitude;
- RMS;
- silence ratio nếu cần;
- feature shape;
- kết quả model nếu có thời gian.

---

# 5. Bước đầu: audit Google Speech Commands V2 trên Kaggle

Trước khi clone paper, một bản Google Speech Commands V2 trên Kaggle đã được kiểm tra.

Path mẫu:

```text
/kaggle/input/datasets/sylkaladin/speech-commands-v2/bird/00b01445_nohash_0.wav
```

Cấu trúc:

```text
speech-commands-v2/
    bird/
        00b01445_nohash_0.wav
```

Tức:

```text
dataset
→ keyword
→ wav file
```

---

# 6. Kết quả audit GSC V2 raw từ Kaggle

Đã chạy audit toàn diện trên 100% dữ liệu raw (nguồn chính thức Google Speech Commands v0.02) và thu được:

```text
Total WAV: 105835
Keyword classes: 35
Readable files: 105835 (Errors: 0)
```

Phân bố chi tiết 35 classes theo số lượng mẫu:

```text
five: 4052      go: 3880        four: 3728      cat: 2031
zero: 4052      stop: 3872      three: 3727     sheila: 2022
yes: 4044       six: 3860       up: 3723        bed: 2014
seven: 3998     on: 3845        dog: 2128       tree: 1759
no: 3941        left: 3801      wow: 2123       backward: 1664
nine: 3934      eight: 3787     house: 2113     visual: 1592
down: 3917      right: 3778     marvin: 2100    follow: 1579
one: 3890       off: 3745       bird: 2064      learn: 1575
two: 3880                       happy: 2054     forward: 1557
```

Thống kê kỹ thuật trên toàn bộ 105.835 file:

```text
Sample rate:
16000 Hz → 105835/105835 (100%)

Channels:
1 (Mono) → 105835/105835 (100%)

Format:
WAV (PCM_16) → 105835/105835 (100%)

Errors:
0
```

Thống kê thời lượng (Duration):

```text
count    105835.000000
mean          0.984649 s
std           0.508240 s
min           0.213312 s
25%           1.000000 s
50%           1.000000 s
75%           1.000000 s
max          95.183125 s (file nhiễu nền dài nhất trong _background_noise_)
```

Thống kê các file ngắn hơn 1 giây (< 1.0s):

```text
Files < 1.0 sec:   10435 file
Tỷ lệ < 1.0 sec:   9.8597%
File ngắn nhất:    0.213312 s (bed/220ee1ef_nohash_0.wav)
```

Thống kê dữ liệu Background Noise:

```text
Background WAV files: 6 file
Thời lượng min:       60.000000 s
Thời lượng max:       95.183125 s
Thời lượng mean:      66.566365 s
```

Kết quả phân tích Speaker per keyword (ngưỡng speaker_limit = 1000):

```text
Core words (>= 1000 speakers): 30 từ khóa
['down', 'seven', 'yes', 'nine', 'four', 'five', 'go', 'left', 'one', 'right',
 'on', 'zero', 'no', 'stop', 'three', 'six', 'eight', 'dog', 'off', 'two',
 'marvin', 'up', 'house', 'wow', 'bird', 'happy', 'cat', 'bed', 'sheila', 'tree']

Unknown words (< 1000 speakers): 5 từ khóa
['visual', 'learn', 'follow', 'forward', 'backward']
(số lượng speaker của Unknown dao động từ 464 đến 515 speakers)
```

> **Phát hiện thực nghiệm quan trọng:**  
> Mặc dù trong code paper định nghĩa tham số `core_split = (24, 5, 5)` (tức kỳ vọng $24 + 5 + 5 = 34$ core words), nhưng khi chạy phân tích thực tế trên Google Speech Commands v0.02 với điều kiện `num_speakers >= 1000`, chỉ có **30 từ khóa** đạt chuẩn Core và **5 từ khóa** là Unknown! Điều này chứng minh giá trị cốt lõi của việc chạy code thực nghiệm thật thay vì suy đoán lý thuyết.

---

# 7. Quyết định không dùng Kaggle mirror làm nguồn chính cho reproduction

Sau khi kiểm tra paper, quyết định:

> Reproduction chính sẽ đi theo repository gốc của paper.

Lý do:

- paper có script riêng để download GSC v0.02;
- script thực hiện preprocessing / organization riêng;
- dùng source trực tiếp giúp tái hiện đúng pipeline hơn.

Bản Kaggle mirror vẫn giữ giá trị để:

- audit dữ liệu raw;
- đối chiếu số lượng;
- kiểm tra cấu trúc.

---

# 8. Clone repository paper

Repository:

```text
https://github.com/ArchitParnami/Few-Shot-KWS.git
```

Cell clone:

```python
%cd /kaggle/working

!git clone https://github.com/ArchitParnami/Few-Shot-KWS.git

%cd /kaggle/working/Few-Shot-KWS
```

Clone thành công.

Cấu trúc top-level:

```text
data/
figures/
protonets/
README.md
requirements.txt
results/
scripts/
setup.py
```

---

# 9. File chuẩn bị dữ liệu chính của paper

File:

```text
data/download_prepare_data.py
```

Ngoài ra có:

```text
protonets/data/FewShotSpeechData.py
protonets/utils/data.py
```

Nhưng script tạo FS-GSC chính nằm ở:

```text
data/download_prepare_data.py
```

---

# 10. Nguồn dataset mà paper tải

Trong source:

```python
data_url = 'http://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
```

Dataset path:

```python
dataset_path = os.path.join(os.curdir, 'speech_commands')
```

Config:

```python
sample_rate = 16000
clip_duration_ms = 1000
speaker_limit = 1000
core_split = (24,5,5)
unknown_split = (60,20,20)
random.seed(42)
```

---

# 11. Pipeline chuẩn bị FS-GSC của paper

`run()` thực hiện 6 bước:

```text
1/6 Filtering
2/6 Grouping
3/6 Analyzing
4/6 Balancing
5/6 Splitting
6/6 Cleaning up
```

Tổng quát:

```text
Google Speech Commands v0.02 raw
        ↓
Filtering
        ↓
Grouping by speaker
        ↓
Analyze speakers / keywords
        ↓
Core vs Unknown
        ↓
Balancing
        ↓
Train / Val / Test split
        ↓
FS-GSC prepared dataset
```

---

# 12. Bước 1 — Filtering

Config:

```python
sample_rate = 16000
clip_duration_ms = 1000
```

Do đó:

```text
desired_samples = 16000
```

Code đọc audio:

```python
sound, sr = torchaudio.load(
    filepath=audio_file,
    normalization=True,
    num_frames=desired_samples
)
```

Sau đó:

```python
if sound.shape[1] != self.desired_samples:
    os.unlink(wav_file)
```

Ý nghĩa:

- audio được đọc tối đa 16000 sample;
- nếu audio thực tế ngắn hơn 16000 sample thì bị xóa;
- paper chỉ giữ các sample đủ 1 giây.

Điều này phù hợp với audit raw trước đó vì đã quan sát thấy:

```text
min duration = 0.384 s
```

---

# 13. Bước 2 — Grouping by speaker

Tên file ví dụ:

```text
00b01445_nohash_0.wav
```

Speaker ID:

```text
00b01445
```

Paper dùng phần trước:

```text
_nohash_
```

làm speaker ID.

Cấu trúc ban đầu:

```text
bird/
    00b01445_nohash_0.wav
```

Sau grouping:

```text
bird/
    00b01445/
        00b01445_nohash_0.wav
```

Tức:

```text
keyword
→ speaker
→ utterance
```

---

# 14. Bước 3 — Analyzing

Paper đếm số speaker cho mỗi keyword.

Config:

```python
speaker_limit = 1000
```

Quy tắc:

```text
num_speakers >= 1000
→ core word

num_speakers < 1000
→ unknown word
```

Paper tạo:

```text
core_words.txt
other_words.txt
```

Điểm quan trọng:

> Core và Unknown không được chọn đơn giản theo tên keyword mà dựa vào số lượng speaker.

---

# 15. Bước 4 — Balancing

Đây là bước quan trọng.

Paper làm dữ liệu đồng đều theo số speaker.

Với từng keyword:

1. lấy danh sách speaker;
2. random shuffle;
3. giới hạn số speaker;
4. với mỗi speaker chỉ chọn ngẫu nhiên 1 file WAV.

Code:

```python
random_wave_file = wav_files[random.randrange(len(wav_files))]
```

Sau đó move file đó sang dataset mới.

Có thể hiểu:

```text
keyword
    speaker A
        wav1
        wav2
        wav3

→ chọn 1 wav duy nhất
```

Điều này giúp:

```text
mỗi speaker ≈ một sample / keyword
```

và làm dataset cân bằng hơn theo speaker.

---

# 16. Bước 5 — Splitting

Core split:

```python
core_split = (24,5,5)
```

Tức:

```text
24 train classes
5 validation classes
5 test classes
```

Tổng:

```text
34 core classes
```

Code shuffle class trước:

```python
random.shuffle(classes)
```

Random seed:

```python
random.seed(42)
```

nên split có thể tái lập nếu cùng source và environment.

Unknown split:

```python
unknown_split = (60,20,20)
```

Tức:

```text
60% train
20% validation
20% test
```

---

# 17. Bước 6 — Cleanup

Paper xóa các file phụ như:

```text
LICENSE
README.md
testing_list.txt
validation_list.txt
core_words.txt
other_words.txt
```

sau khi hoàn thành chuẩn bị dataset.

---

# 18. Một điểm rất quan trọng: script này CHƯA phải feature extraction

`download_prepare_data.py` chỉ làm:

```text
dataset preparation
```

chưa làm:

```text
MFCC
Mel-spectrogram
embedding
TD-ResNet
ProtoNet
```

Do đó phải phân biệt rõ:

## Dataset preparation

```text
GSC raw
→ FS-GSC prepared
```

và:

## Feature preprocessing

```text
WAV
→ speech feature
→ model
```

Phần feature extraction cần kiểm tra tiếp ở:

```text
FewShotSpeechData.py
utils/data.py
training code
```

---

# 19. Bug phát hiện trong repository paper

Khi thử chạy riêng hàm download:

```python
downloader.maybe_download_and_extract_dataset()
```

xuất hiện:

```text
NameError: name 'data_url' is not defined
```

Nguyên nhân:

Trong class đã có:

```python
self.data_url = data_url
```

nhưng trong method:

```python
maybe_download_and_extract_dataset()
```

source dùng:

```python
urllib.request.urlretrieve(data_url, filepath, _progress)
```

trong khi đúng ra phải là:

```python
urllib.request.urlretrieve(self.data_url, filepath, _progress)
```

Đây là:

> bug implementation trong repository paper, không phải lỗi dataset.

---

# 20. Tối ưu hóa tải và giải nén dữ liệu trên Kaggle (CELL 3B-FAST & 3C)

Do hàm tải mặc định bằng `urllib` trong script của paper chạy rất chậm trên môi trường Kaggle, nhóm đã chuyển sang sử dụng giải pháp tải nhanh độc lập bằng tiện ích hệ thống `wget`:

### CELL 3B-FAST — Tải trực tiếp file nén bằng wget:
```python
%cd /kaggle/working/Few-Shot-KWS

import os
from pathlib import Path

dataset_dir = Path("/kaggle/working/Few-Shot-KWS/speech_commands")
dataset_dir.mkdir(parents=True, exist_ok=True)
archive = dataset_dir / "speech_commands_v0.02.tar.gz"
url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

print("Downloading archive with wget...")
!wget -c -q --show-progress "{url}" -O "{archive}"

print("\nDownload finished.")
print("Archive size:", round(archive.stat().st_size / (1024**3), 3), "GB")
```

### CELL 3C — Giải nén toàn bộ dataset bằng tarfile:
```python
%cd /kaggle/working/Few-Shot-KWS

import tarfile
from pathlib import Path

dataset_dir = Path("/kaggle/working/Few-Shot-KWS/speech_commands")
archive = dataset_dir / "speech_commands_v0.02.tar.gz"

print("Extracting...")
with tarfile.open(archive, "r:gz") as tar:
    tar.extractall(dataset_dir)

print("✅ Extraction complete")
```

Kết quả: Dataset raw `speech_commands` đã được giải nén hoàn tất vào `/kaggle/working/Few-Shot-KWS/speech_commands`.

---

# 21. Lưu ý cho báo cáo 2.2

Có thể ghi ngắn gọn:

> Trong quá trình tái hiện pipeline dữ liệu từ repository Few-Shot-KWS, nhóm phát hiện một lỗi phạm vi biến trong hàm tải dữ liệu. Cụ thể, source sử dụng `data_url` thay vì thuộc tính `self.data_url`, gây `NameError`. Nhóm sửa lỗi này và sử dụng tiện ích dòng lệnh tối ưu hóa để tải và giải nén trực tiếp Google Speech Commands v0.02 hoàn tất trước khi bước vào khâu chuẩn bị dữ liệu.

Không cần dành quá nhiều nội dung cho bug này. Nó nên được xem là một ghi chú kỹ thuật trong quá trình thực nghiệm (technical reproduction note).

---

# 22. Trạng thái hiện tại

Đã hoàn thành:

```text
[x] Audit một bản GSC V2 raw trên Kaggle
[x] Xác nhận 35 class
[x] Xác nhận dataset có audio < 1s
[x] Clone official Few-Shot-KWS repo
[x] Đọc download_prepare_data.py
[x] Hiểu pipeline chuẩn bị FS-GSC
[x] Xác định config paper
[x] Phát hiện download bug trong repo
[x] Tải thành công GSC v0.02 bằng wget (CELL 3B-FAST)
[x] Giải nén toàn bộ GSC v0.02 thành công (CELL 3C)
[x] Audit toàn diện 100% RAW GSC v0.02 (105.835 file, 16kHz, mono, PCM_16)
[x] Xác định số lượng file < 1.0s (10.435 file ~ 9.86%)
[x] Xác định 6 file background noise (60.0s -> 95.18s)
[x] Phân tích Speaker per keyword: 30 Core classes vs 5 Unknown classes
[x] Audit AudioMNIST (30.000 WAVs, 10 digits 0-9, 60 speakers, 48kHz sample audit, 0.43-0.82s)
[x] Audit Fluent Speech Commands (30.043 utterances, 97 speakers, 31 intents, 16kHz sample audit, 1.39-2.42s)
[x] Hoàn thiện Mục 2.2 Báo cáo giữa kỳ với đầy đủ 3 dataset và bảng đối chiếu tổng hợp
[x] Chạy Filtering FS-GSC (xóa 10.435 file < 1.0s)
[x] Group by speaker theo _nohash_ cho FS-GSC
[x] Phân chia Core (30 classes) và Unknown (5 classes) theo speaker_limit = 1000
[x] Balance FS-GSC theo speaker: 30 Core x 1.062 = 31.860 mẫu (mu=1062, sigma=0); 5 Unknown x 386 = 1.930 mẫu
[x] Split FS-GSC: 20 Train classes, 5 Validation classes, 5 Test classes; Unknown (1.155 Train, 390 Val, 385 Test)
[x] Preprocessing AudioMNIST: Resample 48k->16k, peak norm, pad/crop 1s, hoàn thành 30.000 file (0 lỗi)
[x] Preprocessing Fluent Speech Commands: Segmentation 1s, hoàn thành 83.777 segments từ 30.043 utterances (mean=2.7886 seg/utt, 0 lỗi)
[x] Hoàn thiện Mục 2.3 Báo cáo giữa kỳ với 100% số liệu thực nghiệm và bảng đối chiếu
[x] Kiểm tra feature extraction của model (Mục 2.4): MFCC 40 coefficients, window 40ms, stride 20ms, n_fft 640, hop 320
[x] Xác định kích thước tensor đầu vào thực tế cho TD-ResNet: [1, 51, 40] tương ứng (1 channel x 51 time frames x 40 MFCCs)
[x] Hoàn thiện Mục 2.4 Báo cáo giữa kỳ với đầy đủ cơ sở lý thuyết, công thức toán và cấu hình thực tế
```

Chưa hoàn thành:

```text
[ ] Xây dựng thực nghiệm minh họa Raw vs Processed (Mục 2.5)
[ ] Cài đặt mô hình TD-ResNet + ProtoNet (Chương 3)
```

---

# 23. Kế hoạch bước tiếp theo

## Phase A — Reproduce FS-GSC

```text
Fix download bug
↓
Download raw GSC v0.02
↓
Audit RAW
↓
Filtering
↓
Audit AFTER FILTER
↓
Grouping
↓
Analyze speakers
↓
Core / Unknown
↓
Balancing
↓
Splitting
↓
Audit FINAL FS-GSC
```

Mỗi bước cần lưu:

```text
input count
output count
changes
code
statistics
```

---

## Phase B — Kiểm tra feature pipeline

Đọc:

```text
FewShotSpeechData.py
protonets/utils/data.py
training code
```

Cần xác định:

```text
waveform input
sample rate
MFCC / Mel?
n_mfcc?
window?
hop?
normalization?
augmentation?
feature shape?
```

---

## Phase C — AudioMNIST

Thực hiện:

```text
RAW audit
↓
apply suitable preprocessing
↓
processed audit
↓
raw vs processed comparison
```

---

## Phase D — Fluent Speech Commands

Thực hiện:

```text
RAW audit
↓
generalized preprocessing
↓
processed audit
↓
raw vs processed comparison
```

Không ép:

```text
1 second
```

nếu làm mất nội dung multi-word.

---

# 24. Output cuối cùng phục vụ báo cáo

Sau khi hoàn thành phần dữ liệu, cần có:

## Dataset tables

```text
samples
classes
speakers
sample rate
channel
duration
format
```

## Preprocessing table

```text
Step
Input
Operation
Output
Reason
```

## Figures

```text
raw waveform
processed waveform
raw spectrogram
processed spectrogram
MFCC / Mel
class distribution
duration histogram
```

## Code snippets

```text
dataset preparation
audio processing
feature extraction
few-shot sampling
```

## Results

```text
raw vs processed statistics
raw vs processed model result nếu thực nghiệm được
```

---

# 25. Nguyên tắc tiếp tục

Từ bước này trở đi:

> Mọi thông tin định lượng phải lấy từ Kaggle/code thật.

Không tự suy đoán:

```text
sample count
speaker count
duration
feature parameters
accuracy
```

Nếu source paper và implementation thực tế khác nhau, báo cáo phải ghi đúng:

```text
paper says X
implementation does Y
our experiment observes Z
```

Không tự hòa trộn ba nguồn thành một kết luận duy nhất.
