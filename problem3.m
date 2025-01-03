R = 32;
p_center = [150, 0, 120];
phi_values = linspace(0, 2 * pi, 37);

% Calculate circle points
circle_points = zeros(length(phi_values), 3);
for i = 1:length(phi_values)
    phi = phi_values(i);
    circle_points(i, :) = p_center + R * [0, cos(phi), sin(phi)];
end
clf;
close all;
% Extract x, y, z coordinates
x = circle_points(:, 1);
y = circle_points(:, 2);
z = circle_points(:, 3);

% Plot the circle
figure;
plot3(x, y, z, '-o', 'LineWidth', 2, 'MarkerSize', 5);
hold on;
grid on;
xlabel('X [mm]');
ylabel('Y [mm]');
zlabel('Z [mm]');
title('3D Circle Plot');
%axis equal;

% Plot center point
plot3(p_center(1), p_center(2), p_center(3), 'r*', 'MarkerSize', 10);
legend('Circle', 'Center Point', 'Location', 'northeast');



%%
fpath = 'joint_angles.xlsx'; % Specifiy path to excel file here
excelToLatexTable(fpath, 'positive', 'problem3-latex.txt');

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
    latexTable = latexTable + "\t\t$q^{(j)}$ & $q_1^{(j)}$ & $q_2^{(j)}$ & $q_3^{(j)}$ & $q_4^{(j)}$ \\\\ ";
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
    latexTable = latexTable + "\\end{table}\n";
    
    % Write the LaTeX table to the output file
    fileID = fopen(outputFile, 'w');
    fprintf(fileID, latexTable);
    fclose(fileID);
    
    disp(['LaTeX table written to ', outputFile]);
end
