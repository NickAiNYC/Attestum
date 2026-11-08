# @reject_reason: forbidden import
import _thread
_thread.start_new_thread(print, (1,))
