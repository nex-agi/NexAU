#!/usr/bin/env python3
"""Fix session status and best scores based on completed experiments."""

import sqlite3
from pathlib import Path
from datetime import datetime

def fix_sessions():
    """Update session status and best scores based on experiments."""
    db_path = Path('data/autotuning.db')
    if not db_path.exists():
        print("❌ Database not found at data/autotuning.db")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        print("🔍 Checking sessions that need updates...")
        
        # Get all sessions
        cursor.execute("""
            SELECT id, name, status, total_experiments, best_score 
            FROM sessions 
            ORDER BY created_at DESC
        """)
        sessions = cursor.fetchall()
        
        updated_count = 0
        
        for session_id, name, status, total_exp, best_score in sessions:
            print(f"\n📋 Checking session: {session_id[:8]}... ({name})")
            
            # Get experiments for this session
            cursor.execute("""
                SELECT status, overall_score 
                FROM experiments 
                WHERE session_id = ?
            """, (session_id,))
            experiments = cursor.fetchall()
            
            if not experiments:
                print(f"   ⚠️  No experiments found")
                continue
            
            print(f"   Found {len(experiments)} experiments")
            for i, (exp_status, exp_score) in enumerate(experiments):
                print(f"   Experiment {i+1}: status={exp_status}, score={exp_score}")
            
            # Calculate correct values
            completed_experiments = [exp for exp in experiments if exp[0].upper() == 'COMPLETED']
            failed_experiments = [exp for exp in experiments if exp[0].upper() == 'FAILED']
            running_experiments = [exp for exp in experiments if exp[0].upper() == 'RUNNING']
            
            total_experiments = len(experiments)
            
            # Determine session status
            if running_experiments:
                new_status = 'RUNNING'
                new_best_score = None
            elif completed_experiments:
                new_status = 'COMPLETED'
                scores = [exp[1] for exp in completed_experiments if exp[1] is not None]
                new_best_score = max(scores) if scores else None
            elif failed_experiments:
                new_status = 'FAILED'
                new_best_score = None
            else:
                new_status = 'PENDING'
                new_best_score = None
            
            print(f"   Current: status={status}, total_exp={total_exp}, best_score={best_score}")
            print(f"   Should be: status={new_status}, total_exp={total_experiments}, best_score={new_best_score}")
            
            # Check if update is needed
            needs_update = (
                status != new_status or 
                total_exp != total_experiments or 
                best_score != new_best_score
            )
            
            if needs_update:
                print(f"   🔧 UPDATING SESSION...")
                
                cursor.execute("""
                    UPDATE sessions 
                    SET status = ?, 
                        total_experiments = ?, 
                        best_score = ?,
                        completed_at = CASE 
                            WHEN ? IN ('COMPLETED', 'FAILED') THEN ? 
                            ELSE completed_at 
                        END
                    WHERE id = ?
                """, (
                    new_status,
                    total_experiments,
                    new_best_score,
                    new_status,
                    datetime.now().isoformat(),
                    session_id
                ))
                
                updated_count += 1
                print(f"   ✅ Updated successfully!")
            else:
                print(f"   ✅ Already correct, no update needed")
        
        conn.commit()
        print(f"\n🎉 Updated {updated_count} sessions")
        
        if updated_count > 0:
            print("\n💡 Now refresh your Streamlit app to see the changes!")
            print("   Click the '🔄 Refresh Data' button in the app")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    fix_sessions()