#import modules

from tkinter import *
import time
import os
 
# Designing window for registration
 
def register():
    global register_screen
    register_screen = Toplevel(main_screen)
    register_screen.title("Register")
    register_screen.geometry("300x250")
 
    global username
    global password
    global username_entry
    global password_entry
    username = StringVar()
    password = StringVar()
 
    Label(register_screen, text="Please enter details below", bg="blue").pack()
    Label(register_screen, text="").pack()
    username_lable = Label(register_screen, text="Username * ")
    username_lable.pack()
    username_entry = Entry(register_screen, textvariable=username)
    username_entry.pack()
    password_lable = Label(register_screen, text="Password * ")
    password_lable.pack()
    password_entry = Entry(register_screen, textvariable=password, show='*')
    password_entry.pack()
    Label(register_screen, text="").pack()
    Button(register_screen, text="Register", width=10, height=1, bg="blue", command = register_user).pack()
 
 
# Designing window for login 
 
def login():
    global login_screen
    login_screen = Toplevel(main_screen)
    login_screen.title("Login")
    login_screen.geometry("300x250")
    Label(login_screen, text="Please enter details below to login").pack()
    Label(login_screen, text="").pack()
 
    global username_verify
    global password_verify
 
    username_verify = StringVar()
    password_verify = StringVar()
 
    global username_login_entry
    global password_login_entry
 
    Label(login_screen, text="Username * ").pack()
    username_login_entry = Entry(login_screen, textvariable=username_verify)
    username_login_entry.pack()
    Label(login_screen, text="").pack()
    Label(login_screen, text="Password * ").pack()
    password_login_entry = Entry(login_screen, textvariable=password_verify, show= '*')
    password_login_entry.pack()
    Label(login_screen, text="").pack()
    Button(login_screen, text="Login", width=10, height=1, command = login_verify).pack()
 
# Implementing event on register button
 
def register_user():
 
    username_info = username.get()
    password_info = password.get()
 
    file = open(username_info, "w")
    file.write(username_info + "\n")
    file.write(password_info)
    file.close()
 
    username_entry.delete(0, END)
    password_entry.delete(0, END)
 
    Label(register_screen, text="Registration Success", fg="green", font=("Times 16 bold")).pack()
 
# Implementing event on login button 
 
def login_verify():
    username1 = username_verify.get()
    password1 = password_verify.get()
    username_login_entry.delete(0, END)
    password_login_entry.delete(0, END)
 
    list_of_files = os.listdir()
    if username1 in list_of_files:
        file1 = open(username1, "r")
        verify = file1.read().splitlines()
        if password1 in verify:
            login_sucess()
 
        else:
            password_not_recognised()
 
    else:
        user_not_found()
 
# Designing popup for login success
 
def login_sucess():

    gui = Tk()

    gui.title("Real-Time Vehicle Verification of Parking Through the Face and License Plate Recognition")

    gui.configure(width = 1920, height = 1080, background = "blue2")

    text = Label(gui, justify = CENTER, text="Real-Time Vehicle Verification of Parking Through the Face and License Plate Recognition", fg = "Black", bg = "dark goldenrod", font = "Times 22 bold")
    text.place(x = 650, y = 100, anchor = CENTER)

    text = Label(gui, justify = CENTER, text="Main Menu", fg = "Black", bg = "light goldenrod", font = "Times 22 bold")
    text.place(x = 650, y = 200, anchor = CENTER)

    button = Button(gui, text = 'Working at the entrance point of parking', fg = "Black", bg = "gold", font = "Times 16 bold", width = 92, command = entrance_working)
    button.place(x = 650, y = 300, anchor = CENTER)

    button = Button(gui, text = 'Working at the exit point of parking', fg = "Black", bg = "gold", font = "Times 16 bold", width = 92, command = exit_working)
    button.place(x = 650, y = 350, anchor = CENTER)

    button = Button(gui, text='Parking record of Entrance', fg = "Black", bg = "gold", font = "Times 16 bold", width = 92, command = entrance_record)
    button.place(x = 650, y = 400, anchor = CENTER)

    button = Button(gui, text='Parking record of Exit', fg = "Black", bg = "gold", font = "Times 16 bold", width = 92, command = exit_record)
    button.place(x = 650, y = 450, anchor = CENTER)

    button = Button(gui, text='Exit from the system', fg = "Black", bg = "gold", font = "Times 16 bold", width = 92, command = gui.destroy)
    button.place(x = 650, y = 500, anchor=CENTER)

    gui.mainloop() 

def entrance_working():
    
    os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Entrance_Video\Entrance_Video.mp4")
    os.system("Entrance_Detection_of_Vehicle.py")
    os.system("Entrance_License_Plate_Recognition.py")
    os.system("Entrance_Face_Detection.py")
    os.system("Encoding_of_QR_Code_and_Generating_Invoice_With_QR_Code.py")
    os.system("Entrance_Play_Sound.py")

def exit_working():
    
    os.system(r"C:\Users\hafiz\Anaconda\Final_Year_Project_Implementation\Exit_Video\Exit_Video.mp4")
    os.system("Exit_Detection_of_Vehicle.py")
    os.system("Exit_License_Plate_Recognition.py")
    os.system("Exit_Face_Detection.py")
    os.system("Exit_Face_Recognition.py")
    os.system("Exit_Decoding_of_QR_Code.py")
    time.sleep(10)
    os.system("Exit_Play_Sound.py")
    
    
def entrance_record():
     os.system("Entrance_Record_Vehicle.py")

def exit_record():
     os.system("Exit_Record_Vehicle.py")
        
# Designing popup for login invalid password
 
def password_not_recognised():
    global password_not_recog_screen
    password_not_recog_screen = Toplevel(login_screen)
    password_not_recog_screen.title("Success")
    password_not_recog_screen.geometry("150x100")
    Label(password_not_recog_screen, text="Invalid Password ").pack()
    Button(password_not_recog_screen, text="OK", command=delete_password_not_recognised).pack()
 
# Designing popup for user not found
 
def user_not_found():
    global user_not_found_screen
    user_not_found_screen = Toplevel(login_screen)
    user_not_found_screen.title("Success")
    user_not_found_screen.geometry("150x100")
    Label(user_not_found_screen, text="User Not Found").pack()
    Button(user_not_found_screen, text="OK", command=delete_user_not_found_screen).pack()
 
# Deleting popups
 
def delete_login_success():
    login_success_screen.destroy()
 
 
def delete_password_not_recognised():
    password_not_recog_screen.destroy()
 
 
def delete_user_not_found_screen():
    user_not_found_screen.destroy()
  
# Designing Main(first) window
 
def main_account_screen():
    global main_screen
    main_screen = Tk()
    main_screen.geometry("300x250")
    main_screen.title("Real-Time Vehicle Verification of Parking Through the Face and License Plate Recognition")
    Label(text="Select Your Choice", width="300", height="2", font=("Times 16 bold")).pack()
    Label(text="").pack()
    Button(text="Login", height="2", width="30", command = login).pack()
    Label(text="").pack()
    Button(text="Register", height="2", width="30", command=register).pack()
 
    main_screen.mainloop()

main_account_screen()