$Text = EnterHereTheConfluenceToken
$Bytes = [System.Text.Encoding]::UTF8.GetBytes($Text)
$EncodedText = [Convert]::ToBase64String($Bytes)
$EncodedText