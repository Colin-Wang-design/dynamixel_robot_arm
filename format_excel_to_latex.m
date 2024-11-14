fpath = 'joint_angles_with_pos.xlsx'; % Specifiy path to excel file here
excelToLatexTable(fpath, 'positive', 'latex_table_output.txt');

function excelToLatexTable(filename, solutionType, outputFile)
    % Reads the Excel file, filters based on solutionType, and creates a LaTeX table.
    % filename: The Excel file with q values
    % solutionType: 'positive' or 'negative'
    % outputFile: The .txt file to save the LaTeX table

    % Load data from Excel
    data = readtable(filename);
    
    % Filter rows based on the selected solution type
    filteredData = data(strcmp(data.Solution, solutionType), :);
    
    % Start constructing the LaTeX table string
    latexTable = "\\begin{table} \n";
    latexTable = latexTable + "%% Generated using the format_excel_to_latex.m matlab script \n"; 
    latexTable = latexTable + "\\centering\n";
    latexTable = latexTable + "\t\\begin{tabular} \n\t\t{l|c|c|c|c} \n";
    latexTable = latexTable + "\t\t\\textbf{$q^{(j)}$} & \\textbf{q1} & \\textbf{q2} & \\textbf{q3} & \\textbf{q4 } \\\\ ";
    latexTable = latexTable + "\\hline\n";

    % Loop through filtered data and format each row
    for i = 1:height(filteredData)
        q = filteredData.Point(i);
        q1 = filteredData.q1(i);
        q2 = filteredData.q2(i);
        q3 = filteredData.q3(i);
        q4 = filteredData.q4(i);
        
        % Append row to the LaTeX table string
        latexTable = latexTable + sprintf("\t\t$q^{%d}$ \t& %.6f\t& %.6f & %.6f & %.6f ", q, q1, q2, q3, q4);
        latexTable = latexTable + "\\\\ \\hline\n";
    end
    
    % End the table
    latexTable = latexTable + "\t\\end{tabular}\n";
    latexTable = latexTable + "\\caption{Robot configurations for points $q^j$ along circle $p^j$.}\n";
    latexTable = latexTable + "\\label{circle-configurations}\n";
    latexTable = latexTable + "\n\\end{table}\n";
    
    % Write the LaTeX table to the output file
    fileID = fopen(outputFile, 'w');
    fprintf(fileID, latexTable);
    fclose(fileID);
    
    disp(['LaTeX table written to ', outputFile]);
end
