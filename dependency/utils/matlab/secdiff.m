function sd = secdiff( datenum1, datenum2 )

sd = etime(datevec(datenum2),datevec(datenum1));

