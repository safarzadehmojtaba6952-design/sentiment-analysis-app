import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
import joblib
import sqlite3
from datetime import datetime
import os

# بررسی وجود فایل‌های مدل
if not os.path.exists('sentiment_model.pkl') or not os.path.exists('vectorizer.pkl'):
    print("⚠️ فایل‌های مدل پیدا نشد! لطفاً ابتدا train_model.py را اجرا کنید.")
    exit()

# بارگذاری مدل
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

print("✅ مدل با موفقیت بارگذاری شد!")

class SentimentApp(App):
    def build(self):
        # ایجاد دیتابیس
        self.init_db()
        
        # طراحی رابط کاربری
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        
        # عنوان
        title = Label(text='🧠 تشخیص احساسات پیامک', font_size=24, size_hint_y=None, height=50)
        layout.add_widget(title)
        
        # زیرنویس
        subtitle = Label(text='جمله‌ی خود را بنویسید تا احساسات آن را تشخیص دهم', font_size=14, size_hint_y=None, height=30)
        layout.add_widget(subtitle)
        
        # ورودی متن
        self.text_input = TextInput(hint_text='مثال: امروز هوای خوبی بود...', multiline=False, font_size=18)
        layout.add_widget(self.text_input)
        
        # دکمه‌ی تشخیص
        btn = Button(text='🔍 تشخیص احساسات', size_hint_y=None, height=50, background_color=(0.2, 0.6, 1, 1))
        btn.bind(on_press=self.predict_sentiment)
        layout.add_widget(btn)
        
        # نمایش نتیجه
        self.result_label = Label(text='نتیجه: ---', font_size=20, size_hint_y=None, height=50)
        layout.add_widget(self.result_label)
        
        # دکمه‌ی نمایش تاریخچه
        history_btn = Button(text='📜 نمایش تاریخچه', size_hint_y=None, height=40)
        history_btn.bind(on_press=self.show_history)
        layout.add_widget(history_btn)
        
        # اسکرول برای تاریخچه
        self.history_scroll = ScrollView(size_hint_y=0.4)
        self.history_label = Label(text='', size_hint_y=None, halign='left', valign='top', font_size=14)
        self.history_label.bind(size=self.history_label.setter('text_size'))
        self.history_scroll.add_widget(self.history_label)
        layout.add_widget(self.history_scroll)
        
        # دکمه‌ی خروج
        exit_btn = Button(text='❌ خروج', size_hint_y=None, height=30, background_color=(1, 0.3, 0.3, 1))
        exit_btn.bind(on_press=self.exit_app)
        layout.add_widget(exit_btn)
        
        return layout
    
    def init_db(self):
        # اتصال به دیتابیس و ایجاد جدول
        conn = sqlite3.connect('sentiments.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                sentiment TEXT,
                timestamp TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ دیتابیس آماده شد!")
    
    def predict_sentiment(self, instance):
        text = self.text_input.text.strip()
        if text == '':
            self.result_label.text = '⚠️ لطفاً یک جمله وارد کنید!'
            return
        
        # پیش‌بینی با مدل
        try:
            text_vector = vectorizer.transform([text])
            sentiment = model.predict(text_vector)[0]
            
            # نمایش نتیجه
            emoji = '😊' if sentiment == 'شاد' else '😢'
            self.result_label.text = f'نتیجه: {sentiment} {emoji}'
            
            # ذخیره در دیتابیس
            conn = sqlite3.connect('sentiments.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sentiments (text, sentiment, timestamp)
                VALUES (?, ?, ?)
            ''', (text, sentiment, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
            conn.close()
            
            # پاک کردن ورودی
            self.text_input.text = ''
            print(f"✅ ذخیره شد: '{text}' => {sentiment}")
            
        except Exception as e:
            self.result_label.text = f'❌ خطا: {str(e)}'
    
    def show_history(self, instance):
        conn = sqlite3.connect('sentiments.db')
        cursor = conn.cursor()
        cursor.execute('SELECT text, sentiment, timestamp FROM sentiments ORDER BY id DESC LIMIT 20')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            self.history_label.text = '📭 هنوز پیامی ذخیره نشده است.\nچند جمله تست بزنید!'
            return
        
        history_text = '📜 آخرین پیام‌ها:\n\n'
        for row in rows:
            emoji = '😊' if row[1] == 'شاد' else '😢'
            text_preview = row[0][:40] + '...' if len(row[0]) > 40 else row[0]
            history_text += f'{emoji} {text_preview}\n   {row[1]} | {row[2]}\n\n'
        
        self.history_label.text = history_text
    
    def exit_app(self, instance):
        App.get_running_app().stop()

# اجرای برنامه
if __name__ == '__main__':
    print("🚀 شروع برنامه...")
    SentimentApp().run()
