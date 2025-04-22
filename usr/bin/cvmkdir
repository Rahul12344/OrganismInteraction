#!/bin/sh
if [ -r /Library/Preferences/Xsan/.xsanrc ]
then
    . /Library/Preferences/Xsan/.xsanrc
fi
if [ x"${XSAN_ALT_BINDIR}" == x ]
then
    XSAN_ALT_BINDIR=/System/Library/Filesystems/acfs.fs/Contents/bin
fi
COMMAND=`basename $0`
${XSAN_ALT_BINDIR}/${COMMAND} "$@"
