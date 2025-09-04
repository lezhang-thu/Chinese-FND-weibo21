# Fake News Detection with User Comments

Relying solely on the textual content of news articles is often insufficient for effective fake news detection (FND), as misleading information is frequently crafted to deceive readers.
However, related discussions on social media platforms such as Weibo and X (formerly Twitter) often provide valuable context through user comments. These responses can serve as a critical signal for evaluating the authenticity of news.

We demonstrate that incorporating user comments into a uni-modal FND pipeline significantly enhances performance. Specifically, we leverage the `[CLS]` token representation obtained from a BERT-based encoder for the target news piece and combine it with the vector representations of associated user comments using a Transformer-based fusion module.

This simple yet effective integration achieves a new state-of-the-art performance on the largest (**high-quality**) Chinese FND dataset, Weibo-21, improving the previous best macro F1-score from **0.943** (BERT-EMO) to **0.97748**.

## Installation
```bash
pip install -U -r requirements.txt
```
## Dataset Preparation
1. Download the MCFEND dataset from [here](https://drive.google.com/drive/folders/1tflhQTkMT_gTTwEw3ESfKS7Sr5w__5u5?usp=sharing).
2. Extract the dataset: `7z x MCFEND.7z`
3. Move the extracted `news.csv` and `social_context.csv` files into the `mcfend` folder of this repository.
4. Run the preprocessing script to extract the part of `social_context.csv` relevant to Weibo-21: `python clean-csv.py`
## Training and Evaluation
- Standard **BERT** fine-tuning (BERT): `bash run.sh`
- **BERT-ST**, which incorporates user comments via a Sentence Transformer: `bash run-comment.sh`
## Results
- BERT

| Seed       | 42     | 43     | 44     | 45     | 46     | Average |
|------------|--------|--------|--------|--------|--------|---------|
|**Macro F1**| 0.9343 | 0.9321 | 0.9299 | 0.9365 | 0.9377 | 0.9341  |
|**Accuracy**| 0.9343 | 0.9321 | 0.9299 | 0.9365 | 0.9380 | 0.93416 |

- BERT-ST

| Seed       | 42     | 43     | 44     | 45     | 46     | Average |
|------------|--------|--------|--------|--------|--------|---------|
|**Macro F1**| 0.981  | 0.9781 | 0.9766 | 0.9796 | 0.9721 | 0.97748 |
|**Accuracy**| 0.981  | 0.9781 | 0.9766 | 0.9796 | 0.9723 | 0.97752 |
