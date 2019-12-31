from flask import Flask
import requests

# Initialize flask application
app = Flask(__name__)

# نقوم بتحميل النموذج و الvectorizer مرة واحدة ليكون جاهزًا للاستخدام
import pickle
# نقوم بفتح ملف النموذج بوضع القراءة rb
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)
# نكرر الأمر لملف الvectorizer
with open('vectorizer.pickle', 'rb') as file:
    vectorizer = pickle.load(file)


# نقوم أيضًا بتعريف دالة تنظيف النصوص، نفس المستخدمة في بناء النموذج
def clean_text(text):
    '''
    يجري هنا تجهيز النص من خلال مسح الرموز و محاولة تبسيطه
    :param text: النص المدخل من نوع str
    :return: النص بعد تجهيزه
    '''

    from re import sub
    text = sub('[^ةجحخهعغفقثصضشسيىبلاآتنمكوؤرإأزدءذئطظ]', ' ', text)
    text = sub(' +', ' ', text)
    text = sub('[آإأ]', 'ا', text)
    text = sub('ة', 'ه', text)

    return text

# سنحتاج إلى معجم صغير لتوضيح نتيجة النموذج، بدلا من إرسالها كرقم، سنرسلها ككلمة
# نعلم مسبقًا أن التصنيف الإيجابي يرمز له بالرقم ١ و السلبي بالرقم ٠
result_dict = {0: 'سلبي', 1: 'إيجابي'}

# Default route to 127.0.0.1:3000
@app.route('/')
def home():
    return "<h1> Hello Data Science Course! </h1>"

# http://127.0.0.1:3000/get_sentiment/أنا سعيد جدا، كانت الرحلة رائعة
@app.route('/get_sentiment/<text>', methods=['GET'])
def get_sentiment(text):
    try:
        # من المتوقع أن نحصل على نص التغريدة

        # تنظيف النص
        text = clean_text(text)
        # ثم سنقوم بتحويل النص إلى أرقام باستخدام الvectorizer
        vector = vectorizer.transform([text])
        # أخيرا ندخل المصفوفة الناتجة إلى النموذج
        result = model.predict(vector)
        # نحول النتيجة و نقوم بإعادتها
        return result_dict[result[0]]
    except:
        return "ERROR"

# Start REST.py on port 3000
if __name__ == "__main__":
   app.run(debug=True, port=3000)

