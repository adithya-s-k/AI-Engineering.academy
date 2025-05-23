# Contributing to AI Engineering Academy

Thank you for your interest in contributing to AI Engineering Academy! This guide will help you understand how to contribute effectively to our educational platform focused on AI engineering skills.

## Our Mission

AI Engineering Academy aims to provide high-quality, practical, and accessible education on AI engineering topics. We emphasize hands-on learning, industry best practices, and practical implementation skills that help learners succeed in real-world AI projects.

## How to Contribute

### Content Contributions

1. **Course Materials**: Help improve existing lessons or create new ones
2. **Code Examples**: Provide clear, well-documented code examples that demonstrate AI engineering concepts
3. **Tutorials**: Create step-by-step guides for specific AI engineering tasks
4. **Case Studies**: Share real-world applications and lessons learned
5. **Exercises**: Develop practice problems that reinforce learning

### Technical Contributions

1. **Platform Improvements**: Enhance our MkDocs-based learning platform
2. **Infrastructure**: Help with deployment, CI/CD, or other technical aspects
3. **Accessibility**: Improve the accessibility of our educational materials
4. **Interactive Features**: Add interactive elements to enhance learning

## Forking and Branching

To contribute to this repository, follow these steps:

1. **Fork the Repository**:
   - Go to the [repository page](https://github.com/adithya-s-k/AI-Engineering.academy).
   - Click the "Fork" button in the top-right corner to create your own copy of the repository.

2. **Clone Your Fork**:
   - Clone your forked repository to your local machine:
     ```bash
     git clone https://github.com/<your-username>/AI-Engineering.academy.git
     cd AI-Engineering.academy
     ```

3. **Create a Branch**:
   - Create a new branch for your changes:
     ```bash
     git checkout -b <branch-name>
     ```
     Use a descriptive branch name, such as `add-new-tutorial` or `fix-bug-in-docs`.

4. **Make Your Changes**:
   - Edit the files and commit your changes:
     ```bash
     git add .
     git commit -m "Describe your changes here"
     ```

5. **Push Your Changes**:
   - Push your branch to your forked repository:
     ```bash
     git push origin <branch-name>
     ```

6. **Create a Pull Request**:
   - Go to the original repository and click "New Pull Request."
   - Select your branch and provide a clear description of your changes.

## Compiling Documentation Locally

Our documentation is built using MkDocs with the Material theme. To compile and preview the documentation locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/adithya-s-k/AI-Engineering.academy.git
   cd AI-Engineering.academy
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Local Server**:
   Start the MkDocs development server:
   ```bash
   mkdocs serve
   ```

4. **Preview the Documentation**:
   Open your browser and navigate to `http://127.0.0.1:8000/` to view the documentation.

5. **Test Changes**:
   Make changes to the documentation files and refresh the browser to see updates in real-time.

6. **Build Static Files** (Optional):
   To generate static files for deployment:
   ```bash
   mkdocs build
   ```

## Contribution Process

### For Content Changes

1. **Identify an Area**: Find content that needs improvement or identify a gap in our curriculum
2. **Create an Issue**: Use our issue templates to propose your content change
3. **Discuss**: Engage with feedback on your proposal
4. **Create a PR**: After discussion, submit your content changes via pull request
5. **Review**: Address feedback from our content reviewers

### For Technical Changes

1. **Identify an Issue**: Find a technical issue or feature to work on
2. **Discuss Implementation**: Share your approach before starting significant work
3. **Create a PR**: Submit your changes with appropriate documentation
4. **Testing**: Ensure your changes work as expected and don't break existing functionality

## Style Guidelines

### Content Style

- **Clear Language**: Write in clear, concise language accessible to non-native English speakers
- **Practical Focus**: Emphasize practical application over pure theory
- **Inclusive Examples**: Use diverse and inclusive examples
- **Visual Learning**: Include diagrams, charts, and visual aids where helpful
- **Proper Citations**: Always cite sources and respect intellectual property

### Code Style

- **Well-documented**: Include comments and docstrings
- **Readable**: Write clean, readable code following established conventions
- **Best Practices**: Demonstrate AI engineering best practices
- **Complete Examples**: Provide full working examples when possible

## Documentation

Our documentation is built using MkDocs with the Material theme. When adding or modifying documentation:

1. Follow the existing structure and formatting
2. Test your changes locally before submitting
3. Ensure proper linking between related content
4. Include appropriate metadata for search and navigation

## Communication

- **Issues**: Use GitHub issues for specific proposals or bugs
- **Discussions**: Use GitHub discussions for broader topics
- **Pull Requests**: For submitting actual changes

## Review Process

All contributions will be reviewed by our team for:

1. **Educational Value**: How well it serves our learning objectives
2. **Technical Accuracy**: Correctness of content and code
3. **Clarity and Quality**: How well it communicates concepts
4. **Integration**: How it fits with existing content

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to adhere to our Code of Conduct, which emphasizes:

- Respectful communication
- Constructive feedback
- Inclusive language and examples
- Professional conduct

Thank you for helping improve AI engineering education!
