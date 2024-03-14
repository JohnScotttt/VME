import sys
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('PyQt6 Window')

        # 创建按钮
        self.button = QPushButton('Click me!', self)
        self.button.clicked.connect(self.on_button_click)

        # 创建垂直布局并将按钮添加到布局中
        layout = QVBoxLayout(self)
        layout.addWidget(self.button)

        self.setLayout(layout)

        # 设置窗口大小固定为900x600
        self.setFixedSize(900, 600)

    def on_button_click(self):
        print('Button clicked!')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec())
