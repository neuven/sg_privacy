import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QTextEdit, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 主布局
        layout = QHBoxLayout()

        # 左侧（数据输入）
        left_layout = QVBoxLayout()
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("输入文本数据...")
        self.image_label = QLabel(self)
        self.image_label.setText("未选择图片")
        self.image_label.setFixedHeight(150)
        btn_load_image = QPushButton("导入图片", self)
        btn_load_image.clicked.connect(self.load_image)
        left_layout.addWidget(self.text_input)
        left_layout.addWidget(btn_load_image)
        left_layout.addWidget(self.image_label)

        # 右侧（用户需求 + 生成按钮 + 结果展示）
        right_layout = QVBoxLayout()
        self.requirement_input = QTextEdit(self)
        self.requirement_input.setPlaceholderText("输入用户需求...")
        btn_generate = QPushButton("生成结果", self)
        btn_generate.clicked.connect(self.generate_result)
        self.result_text = QTextEdit(self)
        self.result_text.setPlaceholderText("这里显示生成的文本结果...")
        self.result_text.setReadOnly(True)
        self.result_image = QLabel(self)
        self.result_image.setText("生成的图片")
        self.result_image.setFixedHeight(150)
        right_layout.addWidget(self.requirement_input)
        right_layout.addWidget(btn_generate)
        right_layout.addWidget(self.result_text)
        right_layout.addWidget(self.result_image)

        # 组合左右布局
        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        self.setLayout(layout)
        self.setWindowTitle('数据输入 & 结果生成')
        self.setGeometry(200, 200, 800, 400)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image_label.setPixmap(QPixmap(file_name).scaled(150, 150))

    def generate_result(self):
        # 这里调用你的生成结果的函数
        data_text = self.text_input.toPlainText()
        user_request = self.requirement_input.toPlainText()

        # 调用处理函数
        generated_text, generated_image_path = self.process_data(data_text, user_request)
        # 更新文本结果
        self.result_text.setPlainText(generated_text)

        # 更新图片结果（如果有）
        if generated_image_path:
            self.result_image.setPixmap(QPixmap(generated_image_path).scaled(150, 150))

    def process_data(self, data, request):
        """
        处理输入数据和用户需求，并返回生成的文本和图片路径
        """
        # 你的处理逻辑，比如：
        generated_text = f"处理文本: {data}\n用户需求: {request}\n生成的文本示例..."
        generated_image_path = "/home/liangxy/pycharm/sg_privacy/data/test_image/50712635678.jpg"  # 假设你生成了一张图片

        return generated_text, generated_image_path


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.show()
    sys.exit(app.exec_())
