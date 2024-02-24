import base64
import oprate
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {'jpg', 'png'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def do_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if request.form['submit_button'] == '10x':
                if file:
                    img_10x = oprate.image_10x(file)
                    image_data, cell_count = oprate.cell_count_10x(img_10x)
                    # 将图像数据转换为 Base64 编码
                    image_base64 = base64.b64encode(image_data.getvalue()).decode()
                    # 渲染模板并传递 Base64 编码的图像和细胞数
                    return render_template('view.html', image_data=image_base64, cell_count=cell_count)
                else:
                    return 'Empty File'
            elif request.form['submit_button'] == '20x':
                if file:
                    img_10x = oprate.image_10x(file)
                    image_data, cell_count = oprate.cell_count_20x(img_10x)
                    # 将图像数据转换为 Base64 编码
                    image_base64 = base64.b64encode(image_data.getvalue()).decode()
                    # 渲染模板并传递 Base64 编码的图像和细胞数
                    return render_template('view.html', image_data=image_base64, cell_count=cell_count)
                else:
                    return 'Empty File'
            elif request.form['submit_button'] == '50x':
                if file:
                    img_10x = oprate.image_10x(file)
                    image_data, cell_count = oprate.cell_count_50x(img_10x)
                    # 将图像数据转换为 Base64 编码
                    image_base64 = base64.b64encode(image_data.getvalue()).decode()
                    # 渲染模板并传递 Base64 编码的图像和细胞数
                    return render_template('view.html', image_data=image_base64, cell_count=cell_count)
                else:
                    return 'Empty File'
        else:
            return 'File not allowed'

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
