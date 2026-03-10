# PolarPandas

A Domain-Specific Language (DSL) for intuitive data cleaning in Python.

## Overview

PolarPandas is a lightweight, Python-native DSL that provides high-level, semantic commands for cleaning tabular data. It abstracts away the complexity of low-level Pandas and Polars APIs, allowing data scientists to focus on analysis rather than data preparation.

**The Problem**: Data scientists spend 60–80% of their working time on data cleaning and preparation tasks. Current tools operate at too low a level of abstraction, requiring multiple method calls to accomplish what is conceptually a single operation.

**The Solution**: Expressive, intent-driven commands that replace repetitive boilerplate code with readable, reusable cleaning pipelines.

## Key Features

- **High-level Cleaning Commands**: Simple, semantic API for common data quality operations
  - `ensure_numeric()` - Type casting
  - `fill_gaps(strategy='mean')` - Imputation
  - `limit_range()` - Outlier control
  - `clean_labels()` - String normalization
  - `parse_dates()` - Date/time parsing

- **Multi-Format Support**: CSV, Excel (.xlsx), and Parquet files
- **Dual Backend**: Works transparently with both Pandas and Polars
- **Type-Driven Cleaning**: Automatic rules based on column classification (Numeric, Categorical, Temporal, Boolean)
- **Readability & Reproducibility**: Clean, maintainable cleaning logic

## Supported Data Quality Problems

- Missing values (null/NaN entries)
- Duplicate records
- Incorrect data types
- Inconsistent formats (dates, names, etc.)
- Outliers and typos

## Target Users

- Junior and mid-level data scientists (primary)
- Data analysts in business, finance, and operations
- Small teams and startups with limited automation infrastructure
- Domain-specific industries: finance, healthcare, academia

## Project Structure

```
polar-pandas-repo/
├── src/
│   ├── lexer/        # Tokenization and lexical analysis
│   ├── parser/       # Syntax parsing and AST generation
│   └── ...           # Additional implementation modules
├── CONTRIBUTING.md   # Development guidelines
├── LICENSE          # License information
└── README.md        # This file
```

## Getting Started

*(Implementation details to be added)*

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for branching strategy, naming conventions, and workflow guidelines.

## Authors

- Ciocanu Ilinca (FAF-241)
- Gafenco Victor (FAF-241)
- Lungu Ilie (FAF-241)
- Mitrofan Alexia (FAF-241)

**Mentor**: Prof. Cojuhari Irina  
**Institution**: Technical University of Moldova, Faculty of Computers, Informatics and Microelectronics

## License

See [LICENSE](LICENSE) file for details.
