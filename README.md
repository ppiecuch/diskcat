## Basic Disk Catalog script

Python script for create file with details/catalog of disk.

`cdcat.py c:/Tmp`

### Reference

1. https://github.com/tsileo/dirtools
2. https://github.com/kirpit/dirtools3

### Example

```
== ROOT FOLDERS ==
------------------
    <DIR> |                                  diskcat |
 363 Byte |                              catalog.bat |
  5.02 Kb |                 Designassets 2020-01.txt |
 705 Byte |                               list.0.bat |
 533 Byte |                               list.1.bat |
 705 Byte |                               list.2.bat |

7 files in 1 folders

== FOLDERS TREE ==
------------------
o
+-- diskcat

== FILES REPORT ==
------------------
 363 Byte |                              catalog.bat | /
  5.02 Kb |                 Designassets 2020-01.txt | /
 705 Byte |                               list.0.bat | /
 533 Byte |                               list.1.bat | /
 705 Byte |                               list.2.bat | /
 38.03 Kb |                                 cdcat.py | /diskcat
 212 Byte |                                README.md | /diskcat
```


Example of disk catalog in Windows:

```
@echo off

set PATH=%PATH%;c:\Python\39

for /f "tokens=1-5*" %%1 in ('vol F:') do (
   set vol=%%6& goto done
)
:done
if "%vol%"=="" goto end

set cat=%vol%.txt

echo -- %vol% --

echo == %vol% == > "%cat%"
echo. >> "%cat%"

vol f: >> "%cat%"
echo. >> "%cat%"

set PYTHONIOENCODING=utf-8

python --version
python diskcat/cdcat.py F: >> "%cat%"

echo.
echo -- Done.

cdr open f:

:end
```
