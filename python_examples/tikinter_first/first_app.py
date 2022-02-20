import tkinter as tk

window = tk.Tk()
window.title('My python scripts')
window.configure(background='black')

text = """
For Debian versions of Linux you have to install it manually by using the following commands.
    For Python 3
sudo apt-get install python3-tk
    For Python 2.7
sudo apt-get install python-tk
Linux distros with yum installer can install tkinter module using the command:
yum install tkinter
Verifying Installation
To verify if you have successfully installed Tkinter, open your Python console and type the following command:
"""


def click():
    entered_text = textentry.get()
    print(entered_text)
    window.destroy()


tk.Label(window, text=text, bg='black', fg='white', font='none 12 bold').grid(row=2,
                                                                              column=0,
                                                                              sticky='e')
textentry = tk.Entry(window, width=20, bg='white')
textentry.grid(row=1, column=0, sticky=tk.W)

tk.Button(window, text='SUBMIT', command=click).grid(row=3, column=0, sticky=tk.W)


window.mainloop()
