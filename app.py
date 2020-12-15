#!/usr/bin/env python
# coding: utf-8



# In[3]:


from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from keras import models
import numpy as np
import wave
import io


app = Flask(__name__)
model = None


def load_model():
    global model
    model = models.load_model('./ESC-50-master/dir\esc50_.03_2.1506_0.5140.hdf5')
    model.summary()
    print('Loaded the model')

@app.route('/')
def index():
    return render_template('./flask_api_index.html')

@app.route('/result', methods=['POST'])
def result():
    # submitした音声が存在したら処理する
    if request.files['sound']:
        # 白黒画像として読み込み
        wav = wave.open(request.files['sound'], "rb")
        # 類似度を出力
        pred = model.predict(wav)

        # render_template('./result.html')
        return render_template('./result.html', title='結果', pred_result=pred)

if __name__ == '__main__':
    load_model()
    app.debug = True
    app.run(host='localhost', port=5000)

