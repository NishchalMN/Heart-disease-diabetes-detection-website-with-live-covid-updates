<?php
	extract($_GET);
	$chunk=108;
	$pos=$count*$chunk;
	$file=fopen("content.txt","r");
	fseek($file,$pos);
	$data=fread($file,$chunk);
	echo $data;
?>