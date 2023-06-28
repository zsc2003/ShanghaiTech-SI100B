import xlrd
data=xlrd.open_workbook("monthly_data_test.xls")

#print(f"包含表单数量:{data.nsheets}")
#print(f"表单名:{data.sheet_names()}")

#sheet=data.sheet_by_index(0)
#print(f"表单名:{sheet.name}")
#print(f"表单索引:{sheet.number}")
#print(f"表单行数:{sheet.nrows}")
#print(f"表单列数:{sheet.ncols}")
#print(f"单元格B2的内容是:{sheet.cell_value(rowx=1,colx=1)}")
#print(f"第一行的内容是:{sheet.row_values(rowx=0)}")
#print(f"第一列的内容是:{sheet.col_values(colx=0)}")

n=data.nsheets
for i in range(n):
    sheet=data.sheet_by_index(i)


import seaborn


