import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

print("🚀 شروع آموزش مدل...")

# ۱. داده‌های آموزشی (همان ۱۰ جمله‌ی ساده)
data = pd.DataFrame({
    'text': [
        'امروز هوای آفتابی و خوبی بود',
        'بهترین روز زندگی‌ام را سپری کردم',
        'دوستم به من هدیه داد',
        'امتحان قبول شدم',
        'تولدم مبارک',
        'دیروز روز بدی داشتم',
        'از دست این ترافیک خسته شدم',
        'دلم برای خانواده تنگ شده',
        'مریض هستم و حال خوبی ندارم',
        'پروژه‌ام را خراب کردم'
    ],
    'label': ['شاد', 'شاد', 'شاد', 'شاد', 'شاد', 
              'غمگین', 'غمگین', 'غمگین', 'غمگین', 'غمگین']
})

print(f"✅ تعداد داده‌های آموزشی: {len(data)}")

# ۲. تبدیل متن به بردار عددی
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['label']

print("✅ تبدیل متن به بردار انجام شد")

# ۳. آموزش مدل
model = MultinomialNB()
model.fit(X, y)

print("✅ مدل با موفقیت آموزش داده شد")

# ۴. تست مدل
test_texts = ["امروز خیلی خوشحالم", "دلم گرفته"]
test_vectors = vectorizer.transform(test_texts)
predictions = model.predict(test_vectors)

for text, pred in zip(test_texts, predictions):
    print(f"📝 '{text}' => احساسات: {pred}")

# ۵. ذخیره‌ی مدل
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ مدل در فایل‌های sentiment_model.pkl و vectorizer.pkl ذخیره شد")
print("🎉 آموزش کامل شد! حالا می‌توانید app.py را اجرا کنید.")
