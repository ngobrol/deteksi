import cv2
import numpy as np 
from moviepy.editor import VideoFileClip
import argparse

#mengambil model
model = "model/Model_data.caffemodel"  
proto = "model/Proto_data.prototxt.txt"

#index deteksi
"""
idx = 0 ==> latar
idx = 1 ==> pesawat
idx = 2 ==> sepeda
idx = 3 ==> burung
idx = 4 ==> perahu
idx = 5 ==> botol
idx = 6 ==> bis
idx = 7 ==> mobil
idx = 8 ==> kucing
idx = 9 ==> kursi
idx = 10 ==> sapi
idx = 11 ==> mejamakan
idx = 12 ==> anjing
idx = 13 ==> kuda
idx = 14 ==> motor
idx = 15 ==> orang
idx = 16 ==> tanamanpot
idx = 17 ==> domba
idx = 18 ==> sofa
idx = 19 ==> kereta
idx = 20 ==> televisi
"""

#Mendefinisikan kelas
kelas = ["latar", "pesawat", "sepeda", "burung", "perahu",
	"botol", "bis", "mobil", "kucing", "kursi", "sapi", "mejamakan",
	"anjing", "kuda", "motor", "orang", "tanamanpot", "domba",
	"sofa", "kereta", "televisi"]

#Random warna
warna = np.random.uniform(0, 255, size=(len(kelas), 3))

#inisialisasi model
net = cv2.dnn.readNetFromCaffe(proto, model)

#Fungsi untuk menjalankan dengan kamera
def camera():
	cam = cv2.VideoCapture(0)

	while True:
		#set awal timer
		timer = cv2.getTickCount()
		#ambil frame
		ret, frame = cam.read()
		#ubah ukuran
		frame = cv2.resize(frame, (1280, 720))

		#ambil ukuran panjang dan lebar
		(h, w) = frame.shape[:2]

		#proses frame
		blob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
		net.setInput(blob)

		#prediksi
		detect = net.forward()

		#hitung fps
		fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

		#loop deteksi beserta menggambar kotak deteksi
		for i in np.arange(0, detect.shape[2]):
			#konfidensi deteksi (buat sebagai threshold)
			conf = detect[0, 0, i, 2]

			#Tampilkan FPS
			cv2.putText(frame, "FPS = %.2f" %fps, (10,60), 
				cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

			#threshold > 0.2
			if conf > 0.2:
				#ambil indeks dan hitung kotak
				idx = int(detect[0, 0, i, 1])
				box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				#ambil label kelas
				label = "{}: {:.2f}%".format(kelas[idx], conf * 100)

				#gambar kotak
				cv2.rectangle(frame, (startX, startY), (endX, endY), warna[idx], 2)

				#set posisi teks dan tampilkan
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y), 
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna[idx], 2)

  		#tampilkan hasil
		cv2.imshow("Tes", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cam.release()
	cv2.destroyAllWindows()

#untuk memproses video. Alur kode hampir sama dengan diatas
def read(image):
	timer = cv2.getTickCount()
	img = cv2.resize(image, (1280, 720))
	img = np.array(img)
	(h, w) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(
		cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
	net.setInput(blob)
	detect = net.forward()

	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

	for i in np.arange(0, detect.shape[2]):
		conf = detect[0, 0, i, 2]
		idx = int(detect[0, 0, i, 1])
		cv2.putText(img, "FPS = %.2f" %fps, (10,60), 
			cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

		#untuk memilih beberapa kelas. Disini saya pilih orang, mobil, sepeda, motor
		if (idx == 2 or idx == 7 or idx == 14 or idx == 15) and conf > 0.2:
			box = detect[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			label = "{}: {:.2f}%".format(kelas[idx], conf * 100)
			cv2.rectangle(img, (startX, startY), (endX, endY), warna[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(img, label, (startX, y), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, warna[idx], 2)

	return img

#Fungsi buat merecord video
def record():
	vid_output = 'output/output.mp4'

	clip1 = VideoFileClip('video/01.mp4')
	vid_clip = clip1.fl_image(read)
	vid_clip.write_videofile(vid_output, audio=False)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--camera', action='store_true')
	parser.add_argument('--video', action='store_true')
	opt = parser.parse_args()

	if opt.camera:
		print('Mode Kamera')
		camera()
	elif opt.video:
		print('Mode Proses Video')
		record()
	else:
		print('Coba jalankan dengan berikut: ')
		print('python deteksi.py --camera untuk menjalankan mode kamera')
		print('python deteksi.py --video untuk menjalankan mode video')
