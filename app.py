# app.py (with AI processor integration)
from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from mcq_processor import MCQProcessor
from ai_mcq_processor import AIMCQProcessor

app = Flask(__name__)
app.secret_key = "mcq_grader_secret_key"
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        try:
            # Get form data
            num_questions = int(request.form['num_questions'])
            options_per_question = int(request.form.get('options_per_question', 4))
            points_correct = float(request.form['points_correct'])
            points_incorrect = float(request.form['points_incorrect'])
            processor_type = request.form.get('processor_type', 'standard')
            
            # Store in session
            session['exam_config'] = {
                'num_questions': num_questions,
                'options_per_question': options_per_question,
                'marking_scheme': {
                    'correct': points_correct,
                    'incorrect': points_incorrect
                },
                'processor_type': processor_type
            }
            
            return redirect(url_for('answer_key'))
        except Exception as e:
            flash(f"Error setting up exam: {str(e)}")
            return redirect(url_for('setup'))
    
    return render_template('setup.html')

@app.route('/answer_key', methods=['GET', 'POST'])
def answer_key():
    if 'exam_config' not in session:
        flash("Please set up exam configuration first")
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        correct_answers = {}
        for i in range(1, session['exam_config']['num_questions'] + 1):
            answer_key = request.form.get(f'q{i}')
            if answer_key:
                correct_answers[i] = answer_key
        
        session['correct_answers'] = correct_answers
        return redirect(url_for('upload'))
    
    return render_template(
        'answer_key.html', 
        num_questions=session['exam_config']['num_questions'],
        options_per_question=session['exam_config']['options_per_question']
    )

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'exam_config' not in session or 'correct_answers' not in session:
        flash("Please set up exam configuration and answer key first")
        return redirect(url_for('setup'))
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            try:
                # Process the uploaded MCQ image based on selected processor type
                if session['exam_config']['processor_type'] == 'ai':
                    # Use AI-enhanced processor
                    processor = AIMCQProcessor()
                    student_answers = processor.process_answer_sheet(
                        file_path,
                        session['exam_config']['num_questions'],
                        session['exam_config']['options_per_question']
                    )
                    
                    # Generate visualization
                    vis_path = processor.visualize_detection(file_path, answers=student_answers)
                    display_path = os.path.basename(vis_path) if vis_path else os.path.basename(file_path)
                else:
                    # Use standard processor
                    processor = MCQProcessor(
                        session['exam_config']['num_questions'],
                        session['exam_config']['options_per_question']
                    )
                    student_answers = processor.process_image(file_path)
                    
                    # Generate enhanced display
                    marked_path = processor.enhance_for_display(file_path, student_answers)
                    display_path = os.path.basename(marked_path) if marked_path else os.path.basename(file_path)
                
                # Grade the exam
                total_score, results = grade_exam(
                    student_answers,
                    session['correct_answers'],
                    session['exam_config']['marking_scheme']
                )
                
                # Store results in session
                session['results'] = {
                    'total_score': total_score,
                    'details': results,
                    'image_path': 'uploads/' + display_path,
                    'student_answers': student_answers
                }
                
                return redirect(url_for('results'))
            except Exception as e:
                flash(f"Error processing image: {str(e)}")
                return redirect(url_for('upload'))
    
    return render_template('upload.html')

@app.route('/results')
def results():
    if 'results' not in session:
        flash("No results to display. Please upload an exam first.")
        return redirect(url_for('upload'))
    
    # Generate a PDF report if requested
    pdf_path = None
    if request.args.get('generate_pdf') == 'true':
        pdf_path = generate_pdf_report(
            session['results'], 
            session['exam_config'],
            session['correct_answers']
        )
    
    return render_template(
        'results.html',
        total_score=session['results']['total_score'],
        details=session['results']['details'],
        image_path=session['results']['image_path'],
        num_questions=session['exam_config']['num_questions'],
        marking_scheme=session['exam_config']['marking_scheme'],
        pdf_path=pdf_path
    )

def grade_exam(student_answers, correct_answers, marking_scheme):
    """Grade the exam based on the marking scheme"""
    total_score = 0
    results = {}
    
    for question, student_answer in student_answers.items():
        question_num = int(question) if isinstance(question, str) else question
        correct = student_answer == correct_answers.get(question_num)
        
        if correct:
            points = marking_scheme['correct']
            result = "Correct"
        else:
            points = marking_scheme['incorrect']
            result = f"Incorrect (Marked {student_answer}, Correct: {correct_answers.get(question_num)})"
            
        total_score += points
        results[question_num] = {
            'marked': student_answer,
            'correct': correct_answers.get(question_num),
            'result': result,
            'points': points
        }
    
    return total_score, results

def generate_pdf_report(results, exam_config, correct_answers):
    """Generate a PDF report of the exam results"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        
        # Create PDF filename
        pdf_filename = f"exam_results_{int(time.time())}.pdf"
        pdf_path = os.path.join('static/results', pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        elements = []
        
        # Add styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        
        # Add title
        elements.append(Paragraph("MCQ Exam Results", title_style))
        elements.append(Paragraph(f"Total Score: {results['total_score']} / {exam_config['num_questions'] * exam_config['marking_scheme']['correct']}", styles['Normal']))
        
        # Create table of results
        data = [['Question', 'Marked Answer', 'Correct Answer', 'Result', 'Points']]
        
        for q_num, result in results['details'].items():
            data.append([
                str(q_num),
                result['marked'],
                result['correct'],
                'Correct' if result['marked'] == result['correct'] else 'Incorrect',
                str(result['points'])
            ])
        
        # Create table
        table = Table(data)
        
        # Style the table
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        # Build PDF
        doc.build(elements)
        
        return pdf_filename
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return None

if __name__ == '__main__':
    import time  # For timestamp in PDF filename
    app.run(debug=True)