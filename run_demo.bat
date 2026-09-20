@echo off
rem One-click demo: build a synthetic 3DGS PLY, convert it two ways.
cd /d "%~dp0"
echo [1/3] generating synthetic 3DGS PLY (demo/demo_3dgs.ply) ...
python make_demo_gs.py -o demo\demo_3dgs.ply || goto :err
echo [2/3] converting with --color dc ...
python gs2ply.py demo\demo_3dgs.ply --color dc -o demo\demo_rgb_dc.ply || goto :err
echo [3/3] converting with --color sh ...
python gs2ply.py demo\demo_3dgs.ply --color sh -o demo\demo_rgb_sh.ply || goto :err
echo.
echo Done. Open demo\demo_rgb_dc.ply in MeshLab / CloudCompare, or run:
echo     python view_ply.py demo\demo_rgb_dc.ply
pause
exit /b 0
:err
echo FAILED - see the error above.
pause
exit /b 1
