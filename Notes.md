


## Stundendaten

https://www.bast.de/videos/2023/zst5060.zip

so they are storing their "Stundendaten" endpoints as:

https://www.bast.de/videos/{year}/zst{BASt-Nr}.zip

, which can be downloaded. Becaause why not store it under videos?


This is the link to display a particular Zählstelle:

https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Daten/2023_1/Jawe2023.html?nn=1819490&cms_detail=5060&cms_map=1

This is also valid:

https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Daten/2022_1/Jawe2022.html?&cms_detail=5060

So it seems like this is the generalized endpoint: 

https://www.bast.de/DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Daten/{year}_1/Jawe{2022}.html?&cms_detail={BASt-Nr}.

The function is called addBKGPoempel, lol. addBKGPoempel(results, "569966.0,5929459.0,red,A1: AD HH-Südost (W) (2217),<div id='markerText'><b>AD HH-Südost (W) (2217; A1)</b><br/>Kfz-Verkehr/Tag: ---<br/>Schwerverkehr/Tag: ---<br/><br/><a href='DE/Verkehrstechnik/Fachthemen/v2-verkehrszaehlung/Daten/2023_1/Jawe2023.html;jsessionid=24C49962C2A038B12DF8E356E12AE069.live11294?nn=1819490&cms_detail=2217&cms_map=0'>weitere Informationen</a></div>");
  
