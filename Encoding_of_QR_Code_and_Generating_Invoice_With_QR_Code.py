import qrcode
import os
from reportlab.lib.utils import ImageReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime

file = open(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_License_Plate\Entrance_License_Plate.txt", "r")
recognition_of_license_plate = file.read()


qr = qrcode.QRCode(
    version=1,
    box_size=5,
    border=1
)

data = recognition_of_license_plate
qr.add_data(data)
qr.make(fit=True)
img = qr.make_image(fill='black', back_color='white')
img.save(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Encoding_of_QR_Code\\' + 'QR_Code' + '.jpg')

today = datetime.today()
current_day = today.strftime("%B %d, %Y")
current_time = today.strftime("%H:%M:%S")

canvas = canvas.Canvas(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Invoice_With_QR_Code\Invoice_With_QR_Code.pdf")
canvas.setPageSize((300, 420))

qr_code = ImageReader(r'C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Encoding_of_QR_Code\QR_Code.jpg')
canvas.setLineWidth(.3)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(20, 390,'Real time vehicle verification through face and ')
canvas.drawString(85, 375, 'license plate recognition')
canvas.line(1, 365, 299, 365)
canvas.drawString(115, 350,'Parking Invoice')
canvas.setFont('Helvetica', 12)
canvas.drawString(7, 310,'Reciept Number: 1')
canvas.drawString(7, 285,'License Plate: '+str(data))
canvas.drawString(160, 310, 'Date: ' + str(current_day))
canvas.drawString(160, 285, 'Entrance Time: ' + str(current_time))
canvas.line(1, 275, 299, 275)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(130, 260,'QR-Code')
canvas.drawImage(qr_code, 100, 140)
canvas.line(1, 130, 299, 130)
canvas.drawString(125, 115,'Instructions')
canvas.setFont('Helvetica', 12)
canvas.drawString(7, 100,'1. Place this slip in front of the scanner at')
canvas.drawString(20, 85,'the time of exit.')
canvas.drawString(7, 70,'2. This invoice is valid for one time only.')
canvas.drawString(7, 55,'3. If the invoice is misplaced. Please immediately')
canvas.drawString(20, 40, 'contact the security.')
canvas.line(1, 30, 299, 30)
canvas.setFont('Helvetica-Bold', 12)
canvas.drawString(110, 15,'Thanks for visitng')
canvas.save()
os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Invoice_With_QR_Code\Invoice_With_QR_Code.pdf")