### Installation and Execution (CPU Only)

1. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```

2. **Install Requirements:**
   ```bash
   pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
   pip install pandas numpy scipy matplotlib
   ```

3. **Run the Program:**
   ```bash
   python A_DHC.py
   ```