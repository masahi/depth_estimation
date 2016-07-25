file = io.open("file_names_shuffled.txt")
file_names = {}
count = 1
for line in file:lines() do
   file_names[count] = 'nyu/' .. line
   print(count, file_names[count])
   count = count + 1
end        
