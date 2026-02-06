# Import the tkinter module and call it tk
import tkinter as tk
from tkinter import filedialog
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# Define a new class LoginWindow that inherits from tk.Tk
class LoginWindow(tk.Tk):
    def __init__(self):
        # Call the constructor of the parent class (tk.Tk)
        super().__init__()
        # Set the title of the window
        self.title("Login")

        # Create a label and entry for the user ID
        self.user_label = tk.Label(self, text="User ID:")
        self.user_entry = tk.Entry(self)
        # Create a label and entry for the password
        self.password_label = tk.Label(self, text="Password:")
        self.password_entry = tk.Entry(self, show="*")
        # Create a button to trigger the login function
        self.login_button = tk.Button(self, text="Login", command=self.login)

        # Pack the labels, entries, and button into the window
        self.user_label.pack()
        self.user_entry.pack()
        self.password_label.pack()
        self.password_entry.pack()
        self.login_button.pack()

    def login(self):
        # Implement the login logic here
        # For demo purposes, just close the login window and open the accuracy checker
        self.withdraw()
        accuracy_checker_window = AccuracyCheckerWindow(self)
        accuracy_checker_window.mainloop()
        # accuracy_checker_window.protocol("WM_DELETE_WINDOW", self.destroy)


# Define a new class AccuracyCheckerWindow that inherits from tk.Toplevel
class AccuracyCheckerWindow(tk.Toplevel):
    def __init__(self, master=None):
        # Call the constructor of the parent class (tk.Toplevel)
        super().__init__(master)
        # Create variables for buttons, labels, and other objects
        self.quit_button = None
        self.classify_data_label = None
        self.train_model_label = None
        self.load_data_label = None
        self.classify_data_button = None
        self.train_model_button = None
        self.load_data_button = None
        self.master = master
        self.filename = ""
        self.dataset = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.classifier = None
        self.y_pred = None
        self.accuracy = None

        self.title("Accuracy Checker")
        self.protocol("WM_DELETE_WINDOW", self.show_sem_non_sems)

        # Call the create_widgets function to populate the window
        self.create_widgets()

    def create_widgets(self):
        # Create a button to load data
        self.load_data_button = tk.Button(self, text="1. Load Data", command=self.load_data)
        self.load_data_button.pack()

        # Create a button to train the model
        self.train_model_button = tk.Button(self, text="2. Train Model", command=self.train_model, state="disabled")
        self.train_model_button.pack()

        # Create a button to classify data
        self.classify_data_button = tk.Button(self, text="3. Classify Data", command=self.classify_data,
                                              state="disabled")
        self.classify_data_button.pack()

        # Create labels to display the status of each action
        self.load_data_label = tk.Label(self, text="1. Load Data: Not Loaded")
        self.load_data_label.pack()
        self.train_model_label = tk.Label(self, text="2. Train Model: Not Trained", state="disabled")
        self.train_model_label.pack()
        self.classify_data_label = tk.Label(self, text="3. Classify Data: Not Classified", state="disabled")
        self.classify_data_label.pack()

        # Create a button to quit the window
        self.quit_button = tk.Button(self, text="Quit", command=self.show_sem_non_sems)
        self.quit_button.pack()

    def load_data(self):
        # Implement the logic to load data here
        self.filename = filedialog.askdirectory(initialdir="/", title="Select folder")
        self.load_data_label.config(text="1. Load Data: " + self.filename)
        # Load the necessary data files
        train_x_file = self.filename + "/Train_x.mat"
        train_y_file = self.filename + "/Train_y.mat"
        test_x_file = self.filename + "/Test_x.mat"
        test_y_file = self.filename + "/Test_y.mat"
        self.X_train = sio.loadmat(train_x_file)['Train_x']
        self.y_train = sio.loadmat(train_y_file)['Train_y'].ravel()
        self.X_test = sio.loadmat(test_x_file)['Test_x']
        self.y_test = sio.loadmat(test_y_file)['Test_y'].ravel()
        self.train_model_button.config(state="normal")

    def train_model(self):
        # Implement the logic to train the model here
        self.classifier = KNeighborsClassifier(n_neighbors=5)
        self.classifier.fit(self.X_train, self.y_train)
        self.train_model_label.config(text="2. Train Model: Completed")
        self.classify_data_button.config(state="normal")

    def classify_data(self):
        # Implement the logic to classify data here
        self.y_pred = self.classifier.predict(self.X_test)
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        self.classify_data_label.config(
            text="3. Classify Data: Completed, Accuracy: {:.2f}%".format(self.accuracy * 100))

    def show_sem_non_sems(self):
        # Close the current window and open the SemNonSemsWindow
        self.destroy()
        sem_non_sems_window = SemNonSemsWindow(self.master)
        sem_non_sems_window.mainloop()
        # sem_non_sems_window.protocol("WM_DELETE_WINDOW", self.destroy)

    def return_to_main(self):
        # Close the current window and open the LoginWindow
        self.destroy()
        self.master.iconify()
        # main_window.protocol("WM_DELETE_WINDOW", self.destroy)


# Define a new class SemNonSemsWindow that inherits from tk.Toplevel
class SemNonSemsWindow(tk.Toplevel):
    def __init__(self, master):
        # Call the constructor of the parent class (tk.Toplevel)
        super().__init__(master)
        # Set the title of the window
        self.title("Sem and Non Sems")

        # Create a label for accuracy percentage
        self.accuracy_label = tk.Label(self, text="Accuracy Percentage:")
        # Create an image for accuracy
        self.accuracy_image = tk.PhotoImage(
            file=r"C:\Users\User\OneDrive\Desktop\Python for KNN\Senior\Figures\accuracy.png")
        # Create a label to display the accuracy image
        self.accuracy_image_label = tk.Label(self, image=self.accuracy_image)
        # Create a button to return to the Accuracy Checker window
        self.return_button = tk.Button(self, text="Return", command=self.return_to_accuracy_checker)

        # Pack the labels, image, and button into the window
        self.accuracy_label.pack()
        self.accuracy_image_label.pack()
        self.return_button.pack()

    def return_to_accuracy_checker(self):
        # Close the current window and open the AccuracyCheckerWindow
        self.destroy()
        self.master.quit()
        # accuracy_checker_window = AccuracyCheckerWindow()
        # accuracy_checker_window.protocol("WM_DELETE_WINDOW", self.destroy)


if __name__ == "__main__":
    # Create an instance of the LoginWindow class and start the main loop
    main_window = LoginWindow()
    main_window.mainloop()