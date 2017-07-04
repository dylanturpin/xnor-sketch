function compressSketchData(file)
% Compress sketch data stored in matfiles by turning them into logical.
m = matfile(file,'Writable',true);
imdb = m.imdb;
imdb.images.data = logical(single(255)-imdb.images.data); % 0/1: bg/fg
m.imdb = imdb;



