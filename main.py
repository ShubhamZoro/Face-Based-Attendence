from flask import Flask, render_template, Response, redirect, url_for, request, flash, session as flask_session
from sqlalchemy import create_engine, Column, Integer, LargeBinary, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import csv
import io
import secrets
from io import StringIO
import base64

app = Flask(__name__)
app.secret_key = '8BYkEfBA6O6donzWlSihBXox7C0sKR6b'  # Required for session management and flashing messages

# Database setup
engine = create_engine('sqlite:///face_recognition.db')
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Database model for storing user face encodings
class UserFace(Base):
    __tablename__ = 'user_faces'
    id = Column(Integer, primary_key=True)
    username = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    encoding = Column(LargeBinary)
    image = Column(LargeBinary)  # To store the uploaded image
    class_id = Column(Integer, ForeignKey('classes.id'))
    reset_token = Column(String)
    is_admin = Column(Boolean, default=False)  # Add this line to track admin users

    # Relationship to attendance records
    attendance_records = relationship("Attendance", back_populates="user")

    # Relationship to class
    student_class = relationship("Class", back_populates="students")

class Class(Base):
    __tablename__ = 'classes'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)

    # Relationship to students
    students = relationship("UserFace", back_populates="student_class")

class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('user_faces.id'))
    timestamp = Column(DateTime, default=datetime.now)
    present = Column(Boolean, default=True)

    # Relationship to user
    user = relationship("UserFace", back_populates="attendance_records")

Base.metadata.create_all(engine)

# Initialize class options if they don't exist
def initialize_classes():
    class_options = ["Computer Science", "Electrical Engineering", "Mechanical Engineering",
                      "Civil Engineering", "Information Technology"]

    for class_name in class_options:
        existing_class = session.query(Class).filter_by(name=class_name).first()
        if not existing_class:
            new_class = Class(name=class_name)
            session.add(new_class)

    session.commit()

# Call this function when the app starts
initialize_classes()

# Function to create an admin user
def create_admin_user():
    admin_email = "admin@example.com"
    admin_password = "adminpassword"
    existing_admin = session.query(UserFace).filter_by(email=admin_email).first()
    if not existing_admin:
        admin_user = UserFace(
            username="Admin",
            email=admin_email,
            password_hash=generate_password_hash(admin_password),
            is_admin=True
        )
        session.add(admin_user)
        session.commit()

# Call this function when the app starts
create_admin_user()

# Video feed generator for capturing frames
def gen_frames(camera):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.template_filter('b64encode')
def b64encode_filter(data):
    if data:
        return base64.b64encode(data).decode('utf-8')
    return ''

# Register page to capture image
@app.route('/register', methods=['GET', 'POST'])
def register():
    # Get all available classes for the dropdown
    classes = session.query(Class).all()

    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        class_id = request.form['class_id']
        file = request.files['image']

        # Check if user already exists
        existing_user = session.query(UserFace).filter_by(email=email).first()
        if existing_user:
            flash("User with this email already exists. Please login instead.")
            return redirect(url_for('login'))

        if file:
            # Read image file
            image_data = file.read()
            # Process the image for face encoding
            image_np = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    encoding = encodings[0]
                    # Hash the password
                    password_hash = generate_password_hash(password)

                    user_face = UserFace(
                        username=username,
                        email=email,
                        password_hash=password_hash,
                        encoding=encoding.tobytes(),
                        image=image_data,
                        class_id=class_id
                    )

                    session.add(user_face)
                    session.commit()
                    flash("Registration successful! Please login.")
                    return redirect(url_for('login'))

            flash("Face not detected. Please try a different image.")

    return render_template('register.html', classes=classes)

# Login page to capture and match face encoding
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form.get('password', '')  # Password is optional for face login
        login_method = request.form.get('login_method', 'face')

        user_face = session.query(UserFace).filter_by(email=email).first()

        if not user_face:
            flash("User not found. Please register first.")
            return redirect(url_for('register'))

        if login_method == 'password':
            # Password login logic
            if check_password_hash(user_face.password_hash, password):
                # Record attendance
                new_attendance = Attendance(user_id=user_face.id)
                session.add(new_attendance)
                session.commit()

                flask_session['user_id'] = user_face.id  # Store user ID in session
                flash(f"Login successful! Welcome, {user_face.username}!")
                if user_face.is_admin:
                    return redirect(url_for('admin'))
                return redirect(url_for('dashboard', user_id=user_face.id))
            else:
                flash("Invalid password. Please try again.")
        else:
            # Face recognition login
            camera = cv2.VideoCapture(0)  # Start the camera when on the login page
            ret, frame = camera.read()
            if ret:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)

                if face_locations:
                    live_encoding = face_recognition.face_encodings(rgb_frame, face_locations)

                    if live_encoding:
                        live_encoding = live_encoding[0]

                        if user_face:
                            stored_encoding = np.frombuffer(user_face.encoding, dtype=np.float64)
                            match = face_recognition.compare_faces([stored_encoding], live_encoding)
                            distance = face_recognition.face_distance([stored_encoding], live_encoding)[0]

                            if match[0] and distance < 0.6:  # Adding a threshold for better security
                                camera.release()  # Release the camera on successful login

                                # Record attendance
                                new_attendance = Attendance(user_id=user_face.id)
                                session.add(new_attendance)
                                session.commit()

                                flask_session['user_id'] = user_face.id  # Store user ID in session
                                flash(f"Login successful! Welcome, {user_face.username}!")
                                if user_face.is_admin:
                                    return redirect(url_for('admin'))
                                return redirect(url_for('dashboard', user_id=user_face.id))
                            else:
                                flash(f"Face verification failed. Try again or use password login.")
                    else:
                        flash("Could not encode face. Please try again or use password login.")
                else:
                    flash("No face detected. Please try again or use password login.")
            else:
                flash("Camera error. Please use password login.")

            camera.release()  # Release the camera if no face is detected or on error

    return render_template('login.html')

# Dashboard page after login
@app.route('/dashboard/<int:user_id>')
def dashboard(user_id):
    user = session.query(UserFace).get(user_id)
    if not user:
        flash("User not found")
        return redirect(url_for('login'))

    # Attendance records
    attendance_records = session.query(Attendance).filter_by(user_id=user_id).all()

    # Class info
    class_info = user.student_class.name if user.student_class else None

    # Convert image to base64 if exists
    image_data = None
    if user.image:
        image_data = base64.b64encode(user.image).decode('utf-8')

    return render_template('dashboard.html',
                           user=user,
                           attendance_records=attendance_records,
                           class_info=class_info,
                           image_data=image_data)

@app.route('/logout')
def logout():
    flask_session.clear()  # or session.pop('user_id') depending on your implementation
    flash("You have been logged out.")
    return redirect(url_for('login'))

# Route for video feed
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)  # Start the camera for video feed
    return Response(gen_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to update user profile photo
@app.route('/update_image/<int:user_id>', methods=['GET', 'POST'])
def update_image(user_id):
    user = session.query(UserFace).get(user_id)
    if not user:
        flash("User not found")
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['image']

        if file:
            # Read image file
            image_data = file.read()
            # Process the image for face encoding
            image_np = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if face_locations:
                encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                if encodings:
                    encoding = encodings[0]

                    # Update user's image and encoding
                    user.image = image_data
                    user.encoding = encoding.tobytes()
                    session.commit()

                    flash("Profile image updated successfully!")
                    return redirect(url_for('dashboard', user_id=user_id))

            flash("Face not detected. Please try a different image.")

    return render_template('update_image.html', user=user)

# Reset password request
@app.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    if request.method == 'POST':
        email = request.form['email']
        user = session.query(UserFace).filter_by(email=email).first()

        if user:
            # Generate a random token
            token = secrets.token_urlsafe(16)
            user.reset_token = token
            session.commit()

            # In a real application, you would send an email with a reset link
            # For this demo, we'll just redirect to the reset page with the token
            flash(f"Password reset requested. In a real application, an email would be sent.")
            return redirect(url_for('reset_password', token=token))
        else:
            flash("Email not found. Please check the email address.")

    return render_template('reset_password_request.html')

# Reset password with token
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    user = session.query(UserFace).filter_by(reset_token=token).first()

    if not user:
        flash("Invalid or expired reset token")
        return redirect(url_for('login'))

    if request.method == 'POST':
        new_password = request.form['password']
        password_hash = generate_password_hash(new_password)

        user.password_hash = password_hash
        user.reset_token = None  # Clear the token after use
        session.commit()

        flash("Password has been reset successfully. Please login with your new password.")
        return redirect(url_for('login'))

    return render_template('reset_password.html', token=token)

# Admin page to view all attendance records by class
@app.route('/admin')
def admin():
    if 'user_id' not in flask_session:
        flash("Please login first.")
        return redirect(url_for('login'))

    user = session.query(UserFace).get(flask_session['user_id'])
    if not user.is_admin:
        flash("You do not have permission to access this page.")
        return redirect(url_for('dashboard', user_id=user.id))

    classes = session.query(Class).all()
    return render_template('admin.html', classes=classes)

# View class attendance
@app.route('/class_attendance/<int:class_id>')
def class_attendance(class_id):
    class_obj = session.query(Class).get(class_id)
    if not class_obj:
        flash("Class not found")
        return redirect(url_for('admin'))

    # Get all students in this class
    students = session.query(UserFace).filter_by(class_id=class_id).all()

    # Get attendance records for all students in this class
    attendance_data = {}
    for student in students:
        attendance_records = session.query(Attendance).filter_by(user_id=student.id).all()
        attendance_data[student.id] = {
            'username': student.username,
            'email': student.email,
            'records': attendance_records
        }

    return render_template('class_attendance.html',
                           class_obj=class_obj,
                           attendance_data=attendance_data)

# Download individual attendance as CSV
@app.route('/download_attendance/<int:user_id>')
def download_attendance(user_id):
    user = session.query(UserFace).get(user_id)
    if not user:
        flash("User not found")
        return redirect(url_for('login'))

    # Get attendance records for this user
    attendance_records = session.query(Attendance).filter_by(user_id=user_id).all()

    # Create CSV data
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Username', 'Email', 'Date', 'Time', 'Present'])

    for record in attendance_records:
        writer.writerow([
            user.username,
            user.email,
            record.timestamp.strftime('%Y-%m-%d'),
            record.timestamp.strftime('%H:%M:%S'),
            'Yes' if record.present else 'No'
        ])

    # Prepare the CSV file for download
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename=attendance_{user.username}.csv'}
    )

# Route to download class attendance as CSV
# Route to download class attendance as CSV
@app.route('/download_class_attendance/<int:class_id>')
def download_class_attendance(class_id):
    class_obj = session.query(Class).get(class_id)
    if not class_obj:
        flash("Class not found")
        return redirect(url_for('admin'))

    # Get all students in this class
    students = session.query(UserFace).filter_by(class_id=class_id).all()

    # Create CSV data
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Username', 'Class', 'Attendance Count'])

    for student in students:
        attendance_count = session.query(Attendance).filter_by(user_id=student.id).count()
        writer.writerow([
            student.username,
            class_obj.name,
            attendance_count
        ])

    # Prepare the CSV file for download
    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment;filename=attendance_{class_obj.name}.csv'}
    )
    
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'user_id' not in flask_session:
        flash("Please login first.")
        return redirect(url_for('login'))

    user = session.query(UserFace).get(flask_session['user_id'])
    if not user or not user.is_admin:
        flash("You do not have permission to access this page.")
        return redirect(url_for('login'))

    file = request.files.get('csv_file')
    if not file or not file.filename.endswith('.csv'):
        flash("Please upload a valid CSV file.")
        return redirect(url_for('admin'))

    csv_data = file.read().decode('utf-8')
    reader = csv.DictReader(StringIO(csv_data))

    success_count = 0
    fail_count = 0

    for row in reader:
        username = row.get('username')
        email = row.get('email')
        password = row.get('password')
        class_name = row.get('class_name')

        if not all([username, email, password, class_name]):
            fail_count += 1
            continue

        if session.query(UserFace).filter_by(email=email).first():
            fail_count += 1
            continue

        student_class = session.query(Class).filter_by(name=class_name).first()
        if not student_class:
            fail_count += 1
            continue

        new_user = UserFace(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            class_id=student_class.id
        )
        session.add(new_user)
        success_count += 1

    session.commit()
    flash(f"{success_count} students registered successfully. {fail_count} failed.")
    return redirect(url_for('admin'))


@app.route('/download_csv_template')
def download_csv_template():
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['username', 'email', 'password', 'class_name'])
    writer.writerow(['John Doe', 'john@example.com', 'password123', 'Computer Science'])

    output.seek(0)
    return Response(
        output,
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment;filename=student_template.csv'}
    )


if __name__ == '__main__':
    app.run(debug=True)
