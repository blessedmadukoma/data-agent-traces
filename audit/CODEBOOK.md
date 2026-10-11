# Codebook: hand audit of the data-reference classification

Version 1, 7 October 2026. Frozen before any label was given (see FREEZE.txt).

## The question

Did the step fail because it referred, by name, to a column, field, key, file, table or collection that does not exist in the data object that it used?

Decide from the error message, the code of the step and the previous steps. Use the schema reference when the error does not show whether the name exists. Do not use what the model intended. The rule depends on whether the name exists.

## Labels

**1. Data-reference error.** All of these are true:

- The step failed with an error.
- The error comes from a reference by name to an item of the data: a column or field of a table, data frame or document; a key of a record that came from the data; a file path; a database table or collection. Access by attribute (`df.month`) counts when the code uses it as a column.
- No item with that name exists in the object that the step used. The object can come directly from a benchmark file or database, or from a transformation of it (filter, group-by, merge, rename). A column that a group-by moved into the index is missing from the result. A wrong path to a file is a missing file.

**2. Not: missing value.** The step looked up a value that is not in the data, for example a fee ID, a merchant name or a date used as a key of a dictionary that the program built from the data. The column or key name exists; the value does not. Exception: in the DAB dataset stockmarket, each ticker is the name of a table, so a query on a ticker table that does not exist is label 1.

**3. Not: other cause.** The failure has a different cause. Examples:

- a syntax error, a type error, a failed conversion of a value;
- a variable, function, method or module that is not defined;
- a key of an object that the program built itself and that does not hold data under that name, for example a dictionary of results;
- a position (`list index out of range`, `iloc`), or an index into `None` or into a number;
- a key that is a number (`KeyError: 0`), unless the data uses numbers as names;
- a rule of the harness or tool: a forbidden function or import, a result variable such as `var_call_...`, a file in `file_storage/`, wrong tool arguments, a time-out;
- an SQL error that is not about a missing table or column (for example `GROUP BY`, a function, a type).

**4. Unclear.** What is shown does not let you decide. Write a short note.

## Examples

| Step | Error | Label |
|---|---|---|
| `payments.groupby('mcc')` on payments.csv | `KeyError: 'mcc'` | 1 (payments.csv has no column `mcc`) |
| `SELECT ProjectName FROM project_info` | `Referenced column "ProjectName" not found` | 1 |
| `pd.read_csv('transactions.csv')` | `FileNotFoundError` | 1 |
| `fees_by_id[1234]`, a dictionary built from fees.json | `KeyError: 1234` | 2 (ID 1234 is a value) |
| `results['total']`, a dictionary the program filled | `KeyError: 'total'` | 3 |
| `json.loads(var_call_abc)` | `NameError` or `FileNotFoundError` on `var_call_abc` | 3 (tool interface) |
| `df.month` where `month` is not a column | `AttributeError: 'DataFrame' object has no attribute 'month'` | 1 |
| `df['date'].str.month` | `AttributeError: 'StringMethods' object has no attribute 'month'` | 3 |
