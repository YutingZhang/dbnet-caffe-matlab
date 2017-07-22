function send_update( task_title, task_status, more_content, attachments )

if ~exist( 'more_content', 'var' )
    more_content = '';
end
if ~exist( 'attachments', 'var' )
    attachments = {};
end

GS = load_global_settings;

subj = sprintf('"%s" is "%s"', task_title, task_status);

sendmail_via_gmail( 'zyt.sender@gmail.com', 'WLcuvVDY', ...
    GS.NOTIFICATION_EMAIL, ['[Update] ' subj], ...
    sprintf('%s\n\nDetails:\n\n%s', subj, more_content), ...
    attachments );



