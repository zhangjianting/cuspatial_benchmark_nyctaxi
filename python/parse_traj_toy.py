import cudf
#input = cudf.Series(["[(1.7,2.6), (1.8,2.7), (1.9,2.8)]"])
	
input=df['locations']
s1 = input.str.replace(["[","(",")","]"], ["","","",""], regex=False)
t1 = s1.str.tokenize(',').str.strip()
d1 = t1.astype('double')
x = d1[0:len(d1):2]
y = d1[1:len(d1):2]