import tkinter as tk
from PIL import Image
import io

import warnings
warnings.filterwarnings('ignore')

from services.servicePreprocess import preprocess
from services.servicePredict import predict, Net

class GUI(tk.Frame):
	def __init__(self, window=None):
		tk.Frame.__init__(self, window)
		self.window = window

		self.canvas_paint = tk.Canvas(self.window, width=300, height=300, bg='white')
		self.canvas_paint.grid(row=0, column=0, columnspan=2, rowspan=2, padx=5, pady=5)
		self.canvas_paint.bind('<B1-Motion>', self.paint)

		self.button_clear = tk.Button(self.window, text='Clear', bg='blue', fg='white', command=self.clear)
		self.button_predict = tk.Button(self.window, text='Predict', bg='green', fg='white', command=self.predict)
		self.button_clear.grid(row=3, column=0, sticky='E', padx=5, pady=5)
		self.button_predict.grid(row=3, column=1, sticky='W', padx=5, pady=5)

		self.label_info = tk.Label(self.window, text='Draw a number')
		self.label_info.grid(row=2, column=0, columnspan=2)

		self.label_result = tk.Label(self.window, text='')
		self.label_result.grid(row=0, column=2, columnspan=2, rowspan=3, padx=100, pady=100)
		self.label_result.config(font =('Helvetica', 48))

		self.label_info2 = tk.Label(self.window, text='Result')
		self.label_info2.grid(row=2, column=2, columnspan=2)

	def paint(self, event):
		size = 6
		x1, y1 = (event.x - size), (event.y - size)
		x2, y2 = (event.x + size), (event.y + size)
		self.canvas_paint.create_oval(x1, y1, x2, y2, fill='black', outline='black')

	def clear(self):
		self.canvas_paint.delete('all')

	def saveCanvas(self):
		self.canvas_paint.update()
		ps = self.canvas_paint.postscript(colormode='color')
		img = Image.open(io.BytesIO(ps.encode('utf-8')))
		return img

	def predict(self):
		image = self.saveCanvas()
		data = preprocess(image)
		output = predict(data)
		self.label_result['text'] = output

if __name__ == '__main__':
	root = tk.Tk()
	root.title('Digit Recognizer')
	root.geometry("600x400")
	window = GUI(root)
	root.mainloop()